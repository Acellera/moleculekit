"""Survey an input structure before building an MD system, and verify the
build result afterwards.

:func:`surveyStructure` runs, in one call, every detection a structure needs
before anything is submitted to a builder: what the structure is (a candidate
RCSB entry recovered by sequence search when the input is a file of unknown
origin), whether it is a membrane protein (entry keywords, when the entry is
known), the full reference sequence of every chain (RCSB entity sequences, a
sequence search, a UniProt accession, or a caller-supplied sequence), the
unresolved stretches (gaps) and sequence mismatches (mutations) against that
reference, and every residue whose bonds and formal charges are unknown. The
results are returned as a printable :class:`SurveyReport` and persisted under
``outdir`` (``input.cif``, ``sequences.json``, ``survey.json``) so later
sessions can act on them. Rerunning is how findings are refined: the resolved
sequences are reused (``sequences.json`` is the source of truth once written),
and each parameter (``pdbid=``, ``sequences=``, ``keep_mutations=``) folds one
decision into the stored state.

:func:`verifyBuildResult` is the closing bracket: it compares a produced
structure against the pre-build input and reports backbone breaks, leftover
ACE/NME-style caps, and per-chain protein residue counts - the checks a job
status cannot make.
"""

import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from moleculekit.molecule import Molecule
from moleculekit.rcsb import rcsbIsMembraneProtein, resolveFullSequences
from moleculekit.tools.modelling import detectBackboneBreaks, detectSequenceGaps
from moleculekit.tools.nonstandard_residues import (
    ChainResidueSpec,
    CovalentLigandSpec,
    LigandSpec,
    ScaffoldSpec,
    detectNonStandardResidues,
)
from moleculekit.uniprot import trimPrecursorSequences, uniprotSequence

logger = logging.getLogger(__name__)

_SPEC_TYPES = {
    ChainResidueSpec: "chain-resident",
    ScaffoldSpec: "scaffold",
    CovalentLigandSpec: "covalent ligand",
    LigandSpec: "free ligand",
}

# Chain-reference sources that a rerun must never overwrite with an RCSB
# lookup: they encode a user decision.
_PROTECTED_SOURCES = {"uniprot", "user"}

_CAP_RESNAMES = ("ACE", "NME", "NHE", "NH2", "NMA")


@dataclass
class SurveyReport:
    """What :func:`surveyStructure` found, printable as a per-topic summary.

    Every field is JSON-serializable; the same content is written to
    ``survey.json`` under the survey's ``outdir``.
    """

    structure: str
    outdir: str
    pdbid: "str | None"
    membrane: "bool | None"
    candidate_pdbid: "str | None"
    chains: dict
    unresolved: list
    gaps: list
    mismatches: list
    skipped_ncaa_chains: list
    nonstandard: list

    def __str__(self) -> str:
        lines = [
            f"Survey of {self.structure} -> {os.path.join(self.outdir, 'input.cif')}"
        ]
        if self.membrane is None:
            lines.append(
                "  membrane     unknown (no PDB id) - ask the user if this is a"
                " membrane protein"
            )
        else:
            lines.append(
                f"  membrane     {self.membrane} (RCSB entry keywords for"
                f" {self.pdbid})"
            )
        if self.pdbid is None:
            if self.candidate_pdbid:
                idents = ", ".join(
                    f"{c}: {m['identity']:.2f}"
                    for c, m in sorted(self.chains.items())
                    if m.get("identity") is not None
                )
                lines.append(
                    f"  identity     candidate {self.candidate_pdbid} ({idents})"
                    " - confirm with the user"
                )
            else:
                lines.append(
                    "  identity     no confident RCSB match - ask the user if"
                    " they know the PDB id"
                )
        for c, m in sorted(self.chains.items()):
            ident = (
                f", identity {m['identity']:.2f}"
                if m.get("identity") is not None
                else ""
            )
            lines.append(
                f"  chain {c}      {m['length']} res ({m['source']}{ident})"
            )
        lines.append(
            "  unresolved   "
            + (", ".join(self.unresolved) if self.unresolved else "(none)")
        )
        if self.skipped_ncaa_chains:
            lines.append(
                "  ncaa chains  "
                + ", ".join(self.skipped_ncaa_chains)
                + " (contain non-canonical residues; gaps cannot be detected)"
            )
        if self.gaps:
            for i, g in enumerate(self.gaps):
                kind = "terminal tail" if g["is_terminal"] else "internal loop"
                lines.append(
                    f"  gaps         [{i}] chain {g['chain']} after resid"
                    f" {g['after_resid']}: {len(g['missing_seq'])} residues"
                    f" ({g['missing_seq']}) - {kind}"
                )
        else:
            lines.append("  gaps         (none)")
        if self.mismatches:
            for m in self.mismatches:
                lines.append(
                    f"  mismatches   chain {m['chain']}:"
                    f" {m['reference']}{m['resid']}{m['insertion']}{m['observed']}"
                )
        else:
            lines.append("  mismatches   (none)")
        if self.nonstandard:
            for n in self.nonstandard:
                lines.append(
                    f"  nonstandard  {n['resname']} {n['resid']}{n['insertion']}"
                    f" chain {n['chain']} ({n['type']})"
                )
        else:
            lines.append("  nonstandard  (none)")
        return "\n".join(lines)


def surveyStructure(
    structure: str,
    outdir: str,
    pdbid: "str | None" = None,
    sequences: "dict | None" = None,
    keep_mutations: bool = False,
    trim: bool = True,
) -> SurveyReport:
    """Survey a structure: everything a builder needs decided, in one call.

    Loads the structure, writes it as ``{outdir}/input.cif``, resolves each
    protein chain's full reference sequence, detects gaps, mismatches and
    non-standard residues, and (when the entry is known) checks the RCSB
    membrane keywords. The resolved sequences are written to
    ``{outdir}/sequences.json`` (plain ``{chain: sequence}``) and the full
    report to ``{outdir}/survey.json``.

    Rerunning refines the stored state instead of redoing it: chains already
    in ``sequences.json`` are not re-resolved (delete the file to start over),
    except that passing a new ``pdbid`` re-resolves every chain whose sequence
    did not come from the user, upgrading search hits to exact entity
    sequences.

    Parameters
    ----------
    structure : str
        A structure file path, or a 4-letter RCSB PDB id.
    outdir : str
        Directory for ``input.cif``, ``sequences.json`` and ``survey.json``.
        Created if missing.
    pdbid : str or None
        The structure's RCSB entry when known. Inferred automatically when
        ``structure`` is itself a PDB id. For a file input, pass the id the
        user confirmed (e.g. the report's ``candidate_pdbid``) to unlock the
        membrane keyword check and exact entity sequences.
    sequences : dict or None
        ``{chain: reference}`` supplied by the user, where each value is
        either a UniProt accession (fetched, and trimmed from the full-length
        precursor to the span the structure covers unless ``trim=False``) or a
        raw one-letter sequence (used verbatim). These chains are never
        overwritten by later reruns.
    keep_mutations : bool
        When True, patch the reference sequences to the residues actually
        observed at mismatch positions before detecting gaps - modelling the
        construct as crystallised rather than reverting it to the reference.
    trim : bool
        Trim user-supplied precursor sequences to the span the structure
        covers (see ``sequences``). Disable only to deliberately extend a
        terminus.

    Returns
    -------
    report : SurveyReport
        The printable survey findings; the same content is in
        ``{outdir}/survey.json``.
    """
    os.makedirs(outdir, exist_ok=True)
    is_id = len(str(structure)) == 4 and not os.path.exists(structure)
    if is_id and pdbid is None:
        pdbid = str(structure)

    mol = Molecule(structure)
    mol.write(os.path.join(outdir, "input.cif"))

    seq_path = os.path.join(outdir, "sequences.json")
    survey_path = os.path.join(outdir, "survey.json")

    seqmap: dict = {}
    meta: dict = {}
    prev_pdbid = None
    if os.path.exists(seq_path):
        with open(seq_path) as fh:
            seqmap = json.load(fh)
        if os.path.exists(survey_path):
            with open(survey_path) as fh:
                prev = json.load(fh)
            prev_pdbid = prev.get("pdbid")
            for c, m in (prev.get("chains") or {}).items():
                if c in seqmap:
                    meta[c] = {
                        k: m.get(k) for k in ("source", "identity", "entity_id")
                    }
    for c in seqmap:
        meta.setdefault(c, {"source": "cached", "identity": None, "entity_id": None})

    observed = {
        c: s
        for c, s in mol.getSequence(
            dict_key="chain", sel="protein", _logger=False
        ).items()
        if s
    }

    # User-supplied references first: they win over everything and survive
    # every rerun.
    if sequences:
        user_chains = []
        for c, value in sequences.items():
            if any(ch.isdigit() for ch in value):  # a UniProt accession
                seqmap[c] = uniprotSequence(value)
                meta[c] = {"source": "uniprot", "identity": None, "entity_id": None}
            else:  # a raw one-letter sequence
                seqmap[c] = value
                meta[c] = {"source": "user", "identity": None, "entity_id": None}
            user_chains.append(c)
        if trim:
            seqmap = trimPrecursorSequences(mol, seqmap, chains=user_chains)

    # Resolve what is still missing - or, when the entry id changed, everything
    # the user did not set themselves.
    if pdbid is not None and prev_pdbid != pdbid:
        need = [
            c
            for c in observed
            if meta.get(c, {}).get("source") not in _PROTECTED_SOURCES
        ]
    else:
        need = [c for c in observed if c not in seqmap]
    if need:
        resolved = resolveFullSequences(mol, pdbid=pdbid)
        for c in need:
            if c in resolved:
                seqmap[c] = resolved[c]["sequence"]
                meta[c] = {
                    k: resolved[c].get(k)
                    for k in ("source", "identity", "entity_id")
                }

    gaps, skipped, mismatches = detectSequenceGaps(mol, seqmap)
    if keep_mutations and mismatches:
        for m in mismatches:
            s = list(seqmap[m["chain"]])
            s[m["ref_index"]] = m["observed"]
            seqmap[m["chain"]] = "".join(s)
        gaps, skipped, mismatches = detectSequenceGaps(mol, seqmap)

    with open(seq_path, "w") as fh:
        json.dump(seqmap, fh, indent=2)

    membrane = None
    if pdbid is not None:
        try:
            membrane = rcsbIsMembraneProtein(pdbid)
        except Exception as e:
            logger.warning(
                f"Could not check the membrane keywords of {pdbid}: {e}. "
                "Reporting membrane as unknown."
            )

    candidate = None
    if pdbid is None:
        entries = [
            m["entity_id"].split("_")[0].upper()
            for m in meta.values()
            if m.get("entity_id")
        ]
        if entries:
            candidate = Counter(entries).most_common(1)[0][0]

    nonstandard = []
    for spec in detectNonStandardResidues(mol):
        r = spec.residue
        nonstandard.append(
            {
                "resname": spec.resname,
                "segid": str(r.segid),
                "chain": str(r.chain),
                "resid": int(r.resid),
                "insertion": str(r.insertion),
                "type": _SPEC_TYPES.get(type(spec), "unknown"),
            }
        )

    report = SurveyReport(
        structure=str(structure),
        outdir=str(outdir),
        pdbid=pdbid,
        membrane=membrane,
        candidate_pdbid=candidate,
        chains={c: {"length": len(seqmap[c]), **meta[c]} for c in seqmap},
        unresolved=[c for c in observed if c not in seqmap],
        gaps=gaps,
        mismatches=mismatches,
        skipped_ncaa_chains=skipped,
        nonstandard=nonstandard,
    )
    with open(survey_path, "w") as fh:
        json.dump(asdict(report), fh, indent=2)
    return report


@dataclass
class VerifyReport:
    """What :func:`verifyBuildResult` found, printable as a per-check summary.

    ``clean`` is True when the produced structure has no backbone breaks and
    no protein residues were lost relative to the input. Leftover caps are
    listed but do not decide ``clean`` on their own: a cap at a genuine chain
    terminus can be legitimate, and a cap patching a break is already caught
    by the break and residue-count checks.
    """

    reference: str
    result: str
    breaks: list
    caps: list
    residues_in: dict
    residues_out: dict

    @property
    def clean(self) -> bool:
        """True when there are no breaks and no protein residues were lost.

        Returns
        -------
        clean : bool
            Whether the two checks passed.
        """
        return not self.breaks and sorted(self.residues_in.values()) == sorted(
            self.residues_out.values()
        )

    def __str__(self) -> str:
        lines = [f"Verified {self.result}", f"  against {self.reference}"]
        if self.breaks:
            for b in self.breaks:
                dist = (
                    "backbone atom missing"
                    if b["distance"] is None
                    else f"{b['distance']:.2f} A"
                )
                lines.append(
                    f"  BREAK {b['segid']}/{b['chain']} resid"
                    f" {b['after_resid']} -> {b['before_resid']}: {dist}"
                )
        else:
            lines.append("  backbone breaks : none")
        if self.caps:
            for c in self.caps:
                lines.append(
                    f"  cap             : {c['resname']} at"
                    f" {c['segid']}/{c['chain']} resid {c['resid']}"
                    " (fine at a real chain terminus; a capped break is not)"
                )
        else:
            lines.append("  caps            : none")
        lines.append(
            f"  protein residues: in {self.residues_in} -> out"
            f" {self.residues_out}"
        )
        lines.append(f"  verdict         : {'CLEAN' if self.clean else 'NOT CLEAN'}")
        return "\n".join(lines)


def verifyBuildResult(reference: str, result: str) -> VerifyReport:
    """Verify a built or prepared structure against the pre-build input.

    Three checks a job status cannot make: backbone continuity of the produced
    structure (an unclosed break is an unmodelled gap or one the builder
    capped), leftover ACE/NME-style caps (legitimate only at a real chain
    terminus), and per-chain protein residue counts against the input (a drop
    means residues were silently discarded).

    Parameters
    ----------
    reference : str
        Path of the pre-build input structure (what the build was asked to
        build).
    result : str
        Path of the produced structure (a build output or a prepared file).

    Returns
    -------
    report : VerifyReport
        The findings; ``report.clean`` says whether every check passed, and
        printing the report gives the per-check summary.
    """
    ref = Molecule(str(reference))
    res = Molecule(str(result))

    breaks = detectBackboneBreaks(res)

    capsel = np.isin(res.resname, _CAP_RESNAMES)
    caps = [
        {"segid": s, "chain": c, "resid": r, "resname": n}
        for s, c, r, n in sorted(
            {
                (str(s), str(c), int(r), str(n))
                for s, c, r, n in zip(
                    res.segid[capsel],
                    res.chain[capsel],
                    res.resid[capsel],
                    res.resname[capsel],
                )
            }
        )
    ]

    def _nres(m):
        seqs = m.getSequence(dict_key="chain", sel="protein", _logger=False)
        return {c: len(s) for c, s in seqs.items() if s}

    return VerifyReport(
        reference=str(reference),
        result=str(result),
        breaks=breaks,
        caps=caps,
        residues_in=_nres(ref),
        residues_out=_nres(res),
    )
