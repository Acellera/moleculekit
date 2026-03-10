from moleculekit.molecule import Molecule
from rdkit import Chem
import numpy as np
import logging

logger = logging.getLogger(__name__)


def rdkitmol_to_molecule(rmol):
    """Converts an RDKit molecule to a Molecule object

    Parameters
    ----------
    rmol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to convert

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> rmol = Chem.MolFromSmiles("C1CCCCC1")
    >>> rmol = Chem.AddHs(rmol)
    >>> AllChem.EmbedMolecule(rmol)
    >>> AllChem.MMFFOptimizeMolecule(rmol)
    >>> mol = Molecule.fromRDKitMol(rmol)
    """
    from rdkit.Chem.rdchem import BondType
    from collections import defaultdict

    bondtype_map = defaultdict(lambda: "un")
    bondtype_map.update(
        {
            BondType.SINGLE: "1",
            BondType.DOUBLE: "2",
            BondType.TRIPLE: "3",
            BondType.QUADRUPLE: "4",
            BondType.QUINTUPLE: "5",
            BondType.HEXTUPLE: "6",
            BondType.AROMATIC: "ar",
        }
    )

    mol = Molecule()
    mol.empty(rmol.GetNumAtoms(), numFrames=rmol.GetNumConformers())
    mol.record[:] = "HETATM"
    mol.resname[:] = rmol.GetProp("_Name") if rmol.HasProp("_Name") else "UNK"
    mol.resid[:] = 1
    mol.viewname = mol.resname[0]

    if rmol.GetNumConformers() != 0:
        mol.coords = np.array(
            np.stack([cc.GetPositions() for cc in rmol.GetConformers()], axis=2),
            dtype=Molecule._dtypes["coords"],
        )

    def get_atomname(mol, element):
        i = 1
        while True:
            name = f"{element}{i}"
            if name not in mol.name:
                return name
            i += 1

    for i, atm in enumerate(rmol.GetAtoms()):
        mol.name[i] = f"{atm.GetSymbol()}_X"  # Rename later
        if atm.HasProp("_Name") and atm.GetProp("_Name") != "":
            mol.name[i] = atm.GetProp("_Name")
        mol.element[i] = atm.GetSymbol()
        mol.formalcharge[i] = atm.GetFormalCharge()
        if atm.HasProp("_TriposPartialCharge"):
            mol.charge[i] = atm.GetPropsAsDict()["_TriposPartialCharge"]
        if atm.HasProp("_TriposAtomType"):
            mol.atomtype[i] = atm.GetPropsAsDict()["_TriposAtomType"]

    for i in range(mol.numAtoms):
        if mol.name[i].endswith("_X"):
            mol.name[i] = get_atomname(mol, mol.element[i])

    for bo in rmol.GetBonds():
        mol.addBond(
            bo.GetBeginAtomIdx(), bo.GetEndAtomIdx(), bondtype_map[bo.GetBondType()]
        )

    return mol


def molecule_to_rdkitmol(
    mol: Molecule,
    sanitize=False,
    kekulize=False,
    assignStereo=True,
    guessBonds=False,
    _logger=True,
):
    """Converts the Molecule to an RDKit molecule

    Parameters
    ----------
    mol : Molecule
        The molecule to convert to an RDKit molecule
    sanitize : bool
        If True the molecule will be sanitized
    kekulize : bool
        If True the molecule will be kekulized
    assignStereo : bool
        If True the molecule will have stereochemistry assigned from its 3D coordinates
    guessBonds : bool
        If True the molecule will have bonds guessed
    _logger : bool
        If True the logger will be used to print information
    """
    from moleculekit.molecule import calculateUniqueBonds
    from rdkit.Chem.rdchem import BondType
    from rdkit.Chem.rdchem import Conformer
    from rdkit import Chem

    if len(set(mol.resid)) != 1:
        raise RuntimeError(
            "The molecule has multiple residues. Please filter it to a single residue first."
        )

    bondtype_map = {
        "1": BondType.SINGLE,
        "2": BondType.DOUBLE,
        "3": BondType.TRIPLE,
        "4": BondType.QUADRUPLE,
        "5": BondType.QUINTUPLE,
        "6": BondType.HEXTUPLE,
        "ar": BondType.AROMATIC,
    }

    _rdmol = Chem.rdchem.RWMol()
    for i in range(mol.numAtoms):
        atm = Chem.rdchem.Atom(mol.element[i])
        atm.SetNoImplicit(
            False
        )  # Set to True to stop rdkit from adding implicit hydrogens
        atm.SetProp("_Name", mol.name[i])
        atm.SetFormalCharge(int(mol.formalcharge[i]))
        atm.SetProp("_TriposAtomType", mol.atomtype[i])
        atm.SetDoubleProp("_TriposPartialCharge", float(mol.charge[i]))
        _rdmol.AddAtom(atm)

    bonds = mol.bonds.copy()
    bondtype = mol.bondtype.copy()
    if guessBonds:
        bonds, bondtype = mol._guessBonds()
    bondtype[bondtype == "un"] = "1"

    if len(mol.bonds) == 0:
        logger.info(
            "No bonds found in molecule. If you want your rdkit molecule to have bonds, use guessBonds=True"
        )

    bonds, bondtypes = calculateUniqueBonds(bonds, bondtype)
    for i in range(bonds.shape[0]):
        _rdmol.AddBond(
            int(bonds[i, 0]),
            int(bonds[i, 1]),
            bondtype_map[bondtypes[i]],
        )

    for f in range(mol.numFrames):
        conf = Conformer(mol.numAtoms)
        for i in range(mol.numAtoms):
            conf.SetAtomPosition(i, mol.coords[i, :, f].tolist())
        _rdmol.AddConformer(conf, assignId=True)

    if sanitize:
        Chem.SanitizeMol(_rdmol)
    if kekulize:
        Chem.Kekulize(_rdmol)
    if assignStereo:
        Chem.AssignStereochemistryFrom3D(_rdmol)

    if _logger:
        logger.info(
            f"Converted Molecule to SmallMol with SMILES: {Chem.MolToSmiles(_rdmol, kekuleSmiles=True)}"
        )

    if mol.numAtoms != _rdmol.GetNumAtoms():
        raise RuntimeError("Number of atoms changed while converting to rdkit molecule")
    _rdmol.SetProp("_Name", mol.resname[0])

    # Return non-editable version
    return Chem.Mol(_rdmol)


def template_residue_from_smiles(
    mol: Molecule,
    sel: str,
    smiles: str,
    sanitizeSmiles: bool = True,
    addHs: bool = False,
    onlyOnAtoms: str | None = None,
    guessBonds: bool = False,
    _logger: bool = True,
):
    """Template a residue from a SMILES string

    This function will assign bonds, bond orders and formal charges to a residue according to a corresponding SMILES string.
    In addition it can also protonate the residue.

    Parameters
    ----------
    mol : Molecule
        The molecule to template the residue in
    sel : str
        The atom selection of the residue which we want to template
    smiles : str
        The SMILES string of the template residue
    sanitizeSmiles : bool
        If True the SMILES string will be sanitized
    addHs : bool
        If True the residue will be protonated
    onlyOnAtoms : str
        If not None, only the atoms in this atom selection will be protonated
    guessBonds : bool
        Set to True to guess bonds for the residue we are templating
    _logger : bool
        If True the logger will be used to print information

    Examples
    --------
    >>> mol = Molecule("3ptb")
    >>> mol.templateResidueFromSmiles("resname BEN", "[NH2+]=C(N)c1ccccc1", addHs=True)
    >>> mol.templateResidueFromSmiles("resname GLY and resid 18", "C(C(=O))N", addHs=True)
    """
    from rdkit.Chem import rdFMCS
    from rdkit import Chem
    import numpy as np

    selidx = mol.atomselect(sel, indexes=True)
    # Check that indexes are all sequential (no gaps)
    if not np.all(np.diff(selidx) == 1):
        raise RuntimeError(
            "The selection contains gaps in the atom indexes. Please select a single molecule residue only."
        )

    residue = mol.copy(sel=selidx)
    if guessBonds:
        residue.guessBonds()
    for field in ("resname", "chain", "segid", "resid", "insertion"):
        if len(set(getattr(residue, field))) != 1:
            raise RuntimeError(
                f"The selection contains multiple {field}s. Please select a single molecule residue only."
            )

    if len(residue.bonds) == 0:
        raise RuntimeError(
            "The selection contains no bonds. Please set the bonds of the residue or guess them with guessBonds=True"
        )

    rmol = residue.toRDKitMol(
        sanitize=False, kekulize=False, assignStereo=False, _logger=_logger
    )
    rmol_smi = Chem.MolFromSmiles(smiles, sanitize=sanitizeSmiles)
    Chem.Kekulize(rmol_smi)

    res = rdFMCS.FindMCS(
        [rmol, rmol_smi],
        bondCompare=rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
    )
    patt = Chem.MolFromSmarts(res.smartsString)
    at1 = list(rmol.GetSubstructMatch(patt))
    at2 = list(rmol_smi.GetSubstructMatch(patt))
    atom_mapping = {j: i for i, j in zip(at1, at2)}

    # Check if any atoms in rmol_smi which are not in at2 are non-hydrogen atoms
    elem = np.array([a.GetSymbol() for a in rmol_smi.GetAtoms()])
    non_matched = np.setdiff1d(range(len(elem)), at2)
    if np.any(elem[non_matched] != "H"):
        raise RuntimeError(
            f"The SMILES template '{smiles}' contains heavy atoms which could not be matched to the residue. "
            "If templating a non-standard amino acid, please don't include the OXT atom in the SMILES string "
            "unless it exists in the structure. For example Glycine should be templated as 'C(C(=O))N' and not 'C(C(=O)O)N'"
        )

    # Transfer the formal charge, bonds and bond orders from rmol_smi to rmol
    for i, j in zip(at1, at2):
        fch = rmol_smi.GetAtomWithIdx(j).GetFormalCharge()
        rmol.GetAtomWithIdx(i).SetFormalCharge(fch)

    for bond in rmol_smi.GetBonds():
        i, j = (
            atom_mapping[bond.GetBeginAtomIdx()],
            atom_mapping[bond.GetEndAtomIdx()],
        )
        btype = bond.GetBondType()
        rbond = rmol.GetBondBetweenAtoms(i, j)
        if rbond is None:
            raise RuntimeError(f"Bond between atoms {i} and {j} not found in residue")
        rbond.SetBondType(btype)

    # Protonate the residue according to the SMILES template
    if addHs:
        if onlyOnAtoms is not None:
            onlyOnAtoms = residue.atomselect(onlyOnAtoms, indexes=True).tolist()

        # Don't sanitize to not lose bond orders
        rmol = Chem.RemoveHs(rmol, sanitize=True)
        rmol = Chem.AddHs(rmol, addCoords=True, onlyOnAtoms=onlyOnAtoms)
        Chem.Kekulize(rmol)  # Sanitization ruins kekulization

    new_residue = Molecule.fromRDKitMol(rmol)
    new_residue.resid[:] = residue.resid[0]
    new_residue.chain[:] = residue.chain[0]
    new_residue.segid[:] = residue.segid[0]
    new_residue.insertion[:] = residue.insertion[0]

    mol.remove(selidx, _logger=False)
    mol.insert(new_residue, selidx[0])


def extend_residue_from_smiles(
    mol: Molecule,
    sel: str,
    extension_smiles: str | None = None,
    target_atom_sel: str | None = None,
    new_smiles: str | None = None,
    sanitizeSmiles: bool = True,
    _logger: bool = True,
):
    """
    Extends a residue by attaching new atoms, preserving original 3D coordinates
    and relaxing new atoms via constrained embedding and MMFF minimization.

    Two modes are supported (mutually exclusive):

    1. Extension SMILES mode: provide ``extension_smiles`` (containing a dummy
       atom ``*`` at the attachment point) together with ``target_atom_sel``
       (the atom on the residue to attach to).

    2. New SMILES mode: provide ``new_smiles`` with the complete SMILES of the
       modified molecule. The function uses Maximum Common Substructure (MCS)
       matching to identify unchanged atoms and generates 3D coordinates for
       the new ones.

    Parameters
    ----------
    mol : Molecule
        The molecule containing the residue to extend
    sel : str
        Atom selection for the residue to extend
    extension_smiles : str, optional
        SMILES of the extension moiety with a dummy atom ``*``
    target_atom_sel : str, optional
        Atom selection of the attachment point (required with extension_smiles)
    new_smiles : str, optional
        Complete SMILES of the modified molecule (alternative to extension_smiles)
    sanitizeSmiles : bool
        If True the SMILES string will be sanitized
    _logger : bool
        If True the logger will be used to print information

    Examples
    --------
    >>> mol = Molecule('3ptb')
    >>> mol.templateResidueFromSmiles("resname BEN", "[NH2+]=C(N)c1ccccc1", addHs=True)
    >>> mol.extendResidueFromSmiles("resname BEN", extension_smiles="*C(C)(C)C", target_atom_sel="resname BEN and name H6")

    Or equivalently using the full modified SMILES:

    >>> mol = Molecule('3ptb')
    >>> mol.templateResidueFromSmiles("resname BEN", "[NH2+]=C(N)c1ccccc1", addHs=True)
    >>> mol.extendResidueFromSmiles("resname BEN", new_smiles="[NH2+]=C(N)c1cc(C(C)(C)C)ccc1")
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    import numpy as np

    if extension_smiles is not None and new_smiles is not None:
        raise ValueError(
            "Cannot specify both 'extension_smiles' and 'new_smiles'. Use one or the other."
        )
    if extension_smiles is None and new_smiles is None:
        raise ValueError(
            "Must specify either 'extension_smiles' (with 'target_atom_sel') or 'new_smiles'."
        )
    if extension_smiles is not None and target_atom_sel is None:
        raise ValueError(
            "'target_atom_sel' is required when using 'extension_smiles'."
        )

    selidx = mol.atomselect(sel, indexes=True)

    if extension_smiles is not None:
        target_atom_idx = mol.atomselect(target_atom_sel, indexes=True)
        if len(target_atom_idx) != 1:
            raise RuntimeError(
                f"The target atom selection contains multiple atoms {target_atom_idx}. Please select a single atom only."
            )
        target_atom_idx = target_atom_idx[0]
        if target_atom_idx not in selidx:
            raise RuntimeError(
                f"The target atom {target_atom_idx} is not in the selection {sel}. Please select a single atom only."
            )
        target_atom_idx = int(np.where(selidx == target_atom_idx)[0][0])

    if not np.all(np.diff(selidx) == 1):
        raise RuntimeError(
            "The selection contains gaps in the atom indexes. Please select a single molecule residue only."
        )

    residue = mol.copy(sel=selidx)
    if residue.numFrames > 1:
        raise RuntimeError(
            "The molecule has multiple frames. This operation is not supported for multiple frames. Please use Molecule.dropFrames(keep=[0]) to keep a single frame"
        )

    for field in ("resname", "chain", "segid", "resid", "insertion"):
        if len(set(getattr(residue, field))) != 1:
            raise RuntimeError(
                f"The selection contains multiple {field}s. Please select a single molecule residue only."
            )

    if len(residue.bonds) == 0:
        raise RuntimeError(
            "The selection contains no bonds. Please set the bonds of the residue or guess them with guessBonds=True"
        )

    if extension_smiles is not None:
        # --- Extension SMILES mode ---
        if residue.element[target_atom_idx] == "H":
            heavy = residue.getNeighbors(target_atom_idx)
            residue.remove(target_atom_idx, _logger=False)
            target_atom_idx = int(heavy[0])

        base_mol = residue.toRDKitMol(
            sanitize=False, kekulize=False, assignStereo=False, _logger=_logger
        )
        base_num_atoms = base_mol.GetNumAtoms()

        ext_mol = Chem.MolFromSmiles(extension_smiles, sanitize=sanitizeSmiles)
        ext_mol = Chem.AddHs(ext_mol)
        Chem.Kekulize(ext_mol)

        dummy_idx = -1
        ext_attach_idx = -1
        bond_order = Chem.rdchem.BondType.SINGLE
        for atom in ext_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx = atom.GetIdx()
                dummy_bond = atom.GetBonds()[0]
                bond_order = dummy_bond.GetBondType()
                ext_attach_idx = dummy_bond.GetOtherAtomIdx(dummy_idx)
                break

        if dummy_idx == -1:
            raise ValueError("Extension SMILES must contain a dummy atom '*'.")

        combined_mol = Chem.CombineMols(base_mol, ext_mol)
        rw_mol = Chem.RWMol(combined_mol)

        new_dummy_idx = dummy_idx + base_num_atoms
        new_ext_attach_idx = ext_attach_idx + base_num_atoms

        rw_mol.AddBond(target_atom_idx, new_ext_attach_idx, order=bond_order)
        rw_mol.RemoveAtom(new_dummy_idx)

        final_mol = rw_mol.GetMol()
        Chem.SanitizeMol(final_mol)

        conf = base_mol.GetConformer()
        coord_map = {}
        for i in range(base_num_atoms):
            coord_map[i] = conf.GetAtomPosition(i)

        alignment_map = [(i, i) for i in range(base_num_atoms)]
        fixed_indices = list(range(base_num_atoms))

    else:
        # --- New SMILES mode (MCS-based) ---
        from rdkit.Chem import rdFMCS

        base_mol = residue.toRDKitMol(
            sanitize=False, kekulize=False, assignStereo=False, _logger=_logger
        )

        new_mol = Chem.MolFromSmiles(new_smiles, sanitize=sanitizeSmiles)
        new_mol = Chem.AddHs(new_mol)
        Chem.Kekulize(new_mol)

        mcs_result = rdFMCS.FindMCS(
            [base_mol, new_mol],
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )
        patt = Chem.MolFromSmarts(mcs_result.smartsString)
        base_matched = list(base_mol.GetSubstructMatch(patt))
        new_matched = list(new_mol.GetSubstructMatch(patt))

        if len(base_matched) == 0:
            raise RuntimeError(
                "Could not find any common substructure between the residue "
                "and the new SMILES. Please check that the new SMILES is a "
                "modified version of the original molecule."
            )

        n_base_heavy = sum(
            1 for a in base_mol.GetAtoms() if a.GetAtomicNum() != 1
        )
        n_matched_heavy = sum(
            1
            for idx in base_matched
            if base_mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
        )
        if n_matched_heavy < n_base_heavy * 0.5:
            logger.warning(
                f"Only {n_matched_heavy}/{n_base_heavy} heavy atoms matched "
                "between residue and new SMILES. The result may be unreliable."
            )

        conf = base_mol.GetConformer()
        coord_map = {}
        matched_new_indices = []
        for base_idx, new_idx in zip(base_matched, new_matched):
            coord_map[new_idx] = conf.GetAtomPosition(base_idx)
            matched_new_indices.append(new_idx)

        final_mol = new_mol
        Chem.SanitizeMol(final_mol)

        alignment_map = [
            (new_idx, base_idx)
            for base_idx, new_idx in zip(base_matched, new_matched)
        ]
        fixed_indices = matched_new_indices

    # Constrained embedding: generate 3D for new atoms, keep matched atoms fixed
    res = AllChem.EmbedMolecule(final_mol, coordMap=coord_map, randomSeed=42)
    if res == -1:
        raise RuntimeError(
            "RDKit failed to embed the molecule. The modification might be too constrained."
        )

    rdMolAlign.AlignMol(final_mol, base_mol, atomMap=alignment_map)

    # MMFF minimization: freeze matched atoms, relax new atoms
    ff_props = AllChem.MMFFGetMoleculeProperties(final_mol)
    ff = AllChem.MMFFGetMoleculeForceField(final_mol, ff_props)

    if ff:
        for i in fixed_indices:
            ff.AddFixedPoint(i)
        ff.Minimize(maxIts=1000)
    else:
        logger.warning(
            "Could not set up MMFF force field. Molecule generated but clashes may exist."
        )

    Chem.Kekulize(final_mol)

    new_residue = Molecule.fromRDKitMol(final_mol)
    new_residue.resname[:] = residue.resname[0]
    new_residue.resid[:] = residue.resid[0]
    new_residue.chain[:] = residue.chain[0]
    new_residue.segid[:] = residue.segid[0]
    new_residue.insertion[:] = residue.insertion[0]

    mol.remove(selidx, _logger=False)
    mol.insert(new_residue, selidx[0])
