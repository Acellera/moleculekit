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
    extension_smiles: str,
    target_atom_sel: str,
    sanitizeSmiles: bool = True,
    _logger: bool = True,
):
    """
    Attaches a moiety to a 3D base molecule, preserving original coordinates
    and relaxing the new moiety to avoid steric clashes.

    Assuming base_mol_3D is your starting molecule with 3D coords and explicit Hs
    And target_idx is the index of your C5 atom

    Attach a t-butyl group (*C(C)(C)C) in 3D
    new_mol_3D = attach_moiety_3D(base_mol_3D, "*C(C)(C)C", target_idx)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    import numpy as np

    selidx = mol.atomselect(sel, indexes=True)
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

    # Check that indexes are all sequential (no gaps)
    if not np.all(np.diff(selidx) == 1):
        raise RuntimeError(
            "The selection contains gaps in the atom indexes. Please select a single molecule residue only."
        )

    residue = mol.copy(sel=selidx)
    # Check that the molecule has a single frame
    if residue.numFrames > 1:
        raise RuntimeError(
            "The molecule has multiple frames. This operation is not supported for multiple frames. Please use Molecule.dropFrames(keep=[0]) to keep a single frame"
        )

    # Check that the molecule has a single residue
    for field in ("resname", "chain", "segid", "resid", "insertion"):
        if len(set(getattr(residue, field))) != 1:
            raise RuntimeError(
                f"The selection contains multiple {field}s. Please select a single molecule residue only."
            )

    # Check that the molecule has bonds
    if len(residue.bonds) == 0:
        raise RuntimeError(
            "The selection contains no bonds. Please set the bonds of the residue or guess them with guessBonds=True"
        )

    # If the target atom is a hydrogen, we need to find the heavy atom it is bonded to
    # and use that as the target atom while removing the hydrogen.
    if residue.element[target_atom_idx] == "H":
        heavy = residue.getNeighbors(target_atom_idx)
        residue.remove(target_atom_idx, _logger=False)
        target_atom_idx = int(heavy[0])

    base_mol = residue.toRDKitMol(
        sanitize=False, kekulize=False, assignStereo=False, _logger=_logger
    )
    base_num_atoms = base_mol.GetNumAtoms()

    # 1. Parse extension, add explicit hydrogens, and find dummy atom
    ext_mol = Chem.MolFromSmiles(extension_smiles, sanitize=sanitizeSmiles)
    ext_mol = Chem.AddHs(ext_mol)
    Chem.Kekulize(ext_mol)

    # Find the dummy atom in the SMILES and it's attachment atom. This is the atom that will be attached to the base molecule.
    dummy_idx = -1
    ext_attach_idx = -1
    bond_order = Chem.rdchem.BondType.SINGLE  # Default fallback
    for atom in ext_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom
            dummy_idx = atom.GetIdx()
            # Get the bond connected to the dummy atom
            dummy_bond = atom.GetBonds()[0]
            bond_order = dummy_bond.GetBondType()

            # Get the index of the atom on the other side of that bond
            ext_attach_idx = dummy_bond.GetOtherAtomIdx(dummy_idx)
            break

    if dummy_idx == -1:
        raise ValueError("Extension SMILES must contain a dummy atom '*'.")

    # 2. Combine and link the molecules
    combined_mol = Chem.CombineMols(base_mol, ext_mol)
    rw_mol = Chem.RWMol(combined_mol)

    new_dummy_idx = dummy_idx + base_num_atoms
    new_ext_attach_idx = ext_attach_idx + base_num_atoms

    rw_mol.AddBond(target_atom_idx, new_ext_attach_idx, order=bond_order)

    # Removing the dummy atom shifts indices, but ONLY for atoms that
    # come after it. Our base_mol atoms (indices 0 to base_num_atoms-1)
    # remain completely unaffected.
    rw_mol.RemoveAtom(new_dummy_idx)

    final_mol = rw_mol.GetMol()
    Chem.SanitizeMol(final_mol)

    # 3. Constrained Embedding (Generate 3D for new atoms, keep old where they are)
    conf = base_mol.GetConformer()
    coord_map = {}
    for i in range(residue.numAtoms):
        coord_map[i] = conf.GetAtomPosition(i)

    # Embed the new molecule. We use a random seed for reproducibility.
    res = AllChem.EmbedMolecule(final_mol, coordMap=coord_map, randomSeed=42)
    if res == -1:
        raise RuntimeError(
            "RDKit failed to embed the molecule. The attachment might be too constrained."
        )

    # Snap final_mol back to the absolute 3D coordinates of base_mol
    # atom_map takes a list of tuples: (probe_atom_index, reference_atom_index)
    alignment_map = [(i, i) for i in range(base_num_atoms)]
    rdMolAlign.AlignMol(final_mol, base_mol, atomMap=alignment_map)

    # 4. Force Field Minimization (Resolve Clashes)
    # Get MMFF94 properties and set up the force field
    ff_props = AllChem.MMFFGetMoleculeProperties(final_mol)
    ff = AllChem.MMFFGetMoleculeForceField(final_mol, ff_props)

    if ff:
        # Freeze all atoms from the original base molecule
        for i in range(base_num_atoms):
            ff.AddFixedPoint(i)

        # Minimize the remaining (new) atoms to relax bonds and resolve clashes
        ff.Minimize(maxIts=1000)
    else:
        print(
            "Warning: Could not set up MMFF force field. Molecule generated but clashes may exist."
        )

    Chem.Kekulize(final_mol)

    # Add only the new atoms to the molecule (and the bond)
    new_residue = Molecule.fromRDKitMol(final_mol)
    mol.remove(selidx, _logger=False)
    mol.insert(new_residue, selidx[0])
