/**
 * Convert MoleculeKit dict to molstar Trajectory without file I/O.
 * Based on Acellera's moleculeToCIF.ts, adapted for system builder.
 */

import { CifCategory, CifField, type CifFrame } from 'molstar/lib/mol-io/reader/cif'
import { ComponentBuilder } from 'molstar/lib/mol-model-formats/structure/common/component'
import { EntityBuilder } from 'molstar/lib/mol-model-formats/structure/common/entity'
import { Column } from 'molstar/lib/mol-data/db'
import { getMoleculeType } from 'molstar/lib/mol-model/structure/model/types'
import { mmCIF_Schema } from 'molstar/lib/mol-io/reader/cif/schema/mmcif'
import { Model, type Trajectory } from 'molstar/lib/mol-model/structure'
import { Coordinates, Time, type Frame } from 'molstar/lib/mol-model/structure/coordinates'
import { Task } from 'molstar/lib/mol-task'
import { MmcifFormat } from 'molstar/lib/mol-model-formats/structure/mmcif'
import { createModels } from 'molstar/lib/mol-model-formats/structure/basic/parser'
import { createBasic } from 'molstar/lib/mol-model-formats/structure/basic/schema'
import {
  PluginStateObject,
  PluginStateTransform,
} from 'molstar/lib/mol-plugin-state/objects'
import { ParamDefinition as PD } from 'molstar/lib/mol-util/param-definition'
import type { MoleculeKitDict } from './types'

export type { MoleculeKitDict }

const BONDTYPE_MAP: Record<string, string> = {
  '1': 'sing',
  '2': 'doub',
  '3': 'trip',
  ar: 'arom',
  am: 'amid',
  du: 'delo',
  un: 'unsp',
  nc: 'ncon',
}

function parseConect(
  mol: MoleculeKitDict,
  sites: { [K in keyof mmCIF_Schema['atom_site']]?: CifField }
): CifCategory {
  const id: string[] = []
  const conn_type_id: string[] = []
  const value_order: string[] = []
  const ptnr1_label_asym_id: string[] = []
  const ptnr1_auth_seq_id: number[] = []
  const ptnr1_label_atom_id: string[] = []
  const ptnr1_label_alt_id: string[] = []
  const ptnr1_PDB_ins_code: string[] = []
  const ptnr2_label_asym_id: string[] = []
  const ptnr2_auth_seq_id: number[] = []
  const ptnr2_label_atom_id: string[] = []
  const ptnr2_label_alt_id: string[] = []
  const ptnr2_PDB_ins_code: string[] = []

  const bonds = mol.bonds
  for (let i = 0; i < bonds.length; i++) {
    const idxA = bonds[i][0]
    const idxB = bonds[i][1]
    // "mc" (moleculekit metal-coordination) → mmCIF metalc, which Mol* maps
    // to BondType.Flag.MetallicCoordination and renders as a dashed bond.
    const bt = mol.bondtype[i]
    const isMetalCoord = bt === 'mc'
    const connType = isMetalCoord ? 'metalc' : 'covale'
    id.push(`${connType}${i + 1}`)
    conn_type_id.push(connType)
    value_order.push(isMetalCoord ? 'sing' : (BONDTYPE_MAP[bt] || 'sing'))

    ptnr1_label_asym_id.push(sites.label_asym_id!.str(idxA))
    ptnr1_auth_seq_id.push(sites.auth_seq_id!.int(idxA))
    ptnr1_label_atom_id.push(sites.label_atom_id!.str(idxA))
    ptnr1_label_alt_id.push(sites.label_alt_id!.str(idxA))
    ptnr1_PDB_ins_code.push(sites.pdbx_PDB_ins_code!.str(idxA))

    ptnr2_label_asym_id.push(sites.label_asym_id!.str(idxB))
    ptnr2_auth_seq_id.push(sites.auth_seq_id!.int(idxB))
    ptnr2_label_atom_id.push(sites.label_atom_id!.str(idxB))
    ptnr2_label_alt_id.push(sites.label_alt_id!.str(idxB))
    ptnr2_PDB_ins_code.push(sites.pdbx_PDB_ins_code!.str(idxB))
  }

  const struct_conn: Partial<CifCategory.Fields<mmCIF_Schema['struct_conn']>> = {
    id: CifField.ofStrings(id),
    conn_type_id: CifField.ofStrings(conn_type_id),
    pdbx_value_order: CifField.ofStrings(value_order),
    ptnr1_label_asym_id: CifField.ofStrings(ptnr1_label_asym_id),
    ptnr1_auth_seq_id: CifField.ofNumbers(ptnr1_auth_seq_id),
    ptnr1_label_atom_id: CifField.ofStrings(ptnr1_label_atom_id),
    pdbx_ptnr1_label_alt_id: CifField.ofStrings(ptnr1_label_alt_id),
    pdbx_ptnr1_PDB_ins_code: CifField.ofStrings(ptnr1_PDB_ins_code),
    ptnr2_label_asym_id: CifField.ofStrings(ptnr2_label_asym_id),
    ptnr2_auth_seq_id: CifField.ofNumbers(ptnr2_auth_seq_id),
    ptnr2_label_atom_id: CifField.ofStrings(ptnr2_label_atom_id),
    pdbx_ptnr2_label_alt_id: CifField.ofStrings(ptnr2_label_alt_id),
    pdbx_ptnr2_PDB_ins_code: CifField.ofStrings(ptnr2_PDB_ins_code),
  }

  return CifCategory.ofFields('struct_conn', struct_conn)
}

function getAtomSite(
  mol: MoleculeKitDict,
  label_entity_id: string[],
  terIndices: Set<number>
): { [K in keyof mmCIF_Schema['atom_site'] | 'partial_charge']?: CifField } {
  const numAtoms = mol.numAtoms
  const pdbx_PDB_model_num = CifField.ofStrings(new Array(numAtoms).fill('1'))
  const auth_asym_id = CifField.ofStrings(mol.chain)
  const auth_seq_id = CifField.ofNumbers(mol.resid)
  const pdbx_PDB_ins_code = CifField.ofStrings(mol.insertion)
  const auth_atom_id = CifField.ofStrings(mol.name)
  const auth_comp_id = CifField.ofStrings(mol.resname)
  const id = CifField.ofStrings(mol.serial.map(String))

  let currAsymId = auth_asym_id.str(0)
  let currSeqId = auth_seq_id.int(0)
  let currInsCode = pdbx_PDB_ins_code.str(0)
  let currLabelAsymId = currAsymId
  let currLabelSeqId = currSeqId

  const asymIdCounts = new Map<string, number>()
  const atomIdCounts = new Map<string, number>()
  const labelAsymIds: string[] = []
  const labelAtomIds: string[] = []
  const labelSeqIds: number[] = []

  let hasInsCode = false
  for (let i = 0; i < numAtoms; i++) {
    if (pdbx_PDB_ins_code.str(i) !== '') {
      hasInsCode = true
      break
    }
  }

  for (let i = 0; i < numAtoms; i++) {
    const asymId = auth_asym_id.str(i)
    const seqId = auth_seq_id.int(i)
    const insCode = pdbx_PDB_ins_code.str(i)
    let atomId = auth_atom_id.str(i)

    if (currAsymId !== asymId) {
      atomIdCounts.clear()
      currAsymId = asymId
      currSeqId = seqId
      currInsCode = insCode
      currLabelAsymId = asymId
      currLabelSeqId = seqId
    } else if (currSeqId !== seqId) {
      atomIdCounts.clear()
      if (currSeqId === currLabelSeqId) {
        currLabelSeqId = seqId
      } else {
        currLabelSeqId += 1
      }
      currSeqId = seqId
      currInsCode = insCode
    } else if (currInsCode !== insCode) {
      atomIdCounts.clear()
      currInsCode = insCode
      currLabelSeqId += 1
    }

    if (asymIdCounts.has(asymId)) {
      if (terIndices.has(i)) {
        const count = (asymIdCounts.get(asymId) || 0) + 1
        asymIdCounts.set(asymId, count)
        currLabelAsymId = `${asymId}_${count}`
      }
    } else {
      asymIdCounts.set(asymId, 0)
    }
    labelAsymIds[i] = currLabelAsymId

    if (atomIdCounts.has(atomId)) {
      const count = (atomIdCounts.get(atomId) || 0) + 1
      atomIdCounts.set(atomId, count)
      atomId = `${atomId}_${count}`
    } else {
      atomIdCounts.set(atomId, 0)
    }
    labelAtomIds[i] = atomId

    if (hasInsCode) {
      labelSeqIds[i] = currLabelSeqId
    }
  }

  // Topology is built from frame 0. Slicing with Float32Array gives back
  // a Float32Array; CifField.ofNumbers accepts ArrayLike<number>.
  const coords = mol.coords as ArrayLike<number> & { slice(start: number, end?: number): ArrayLike<number> }
  const label_seq_id = hasInsCode
    ? CifField.ofColumn(Column.ofIntArray(labelSeqIds))
    : CifField.ofUndefined(numAtoms, Column.Schema.int)

  return {
    auth_asym_id,
    auth_atom_id,
    auth_comp_id,
    auth_seq_id,
    B_iso_or_equiv: CifField.ofNumbers(mol.beta),
    Cartn_x: CifField.ofNumbers(coords.slice(0 * numAtoms, 1 * numAtoms)),
    Cartn_y: CifField.ofNumbers(coords.slice(1 * numAtoms, 2 * numAtoms)),
    Cartn_z: CifField.ofNumbers(coords.slice(2 * numAtoms, 3 * numAtoms)),
    group_PDB: CifField.ofStrings(mol.record),
    id,
    label_alt_id: CifField.ofStrings(mol.altloc),
    label_asym_id: CifField.ofColumn(Column.ofStringArray(labelAsymIds)),
    label_atom_id: CifField.ofColumn(Column.ofStringArray(labelAtomIds)),
    label_comp_id: auth_comp_id,
    label_seq_id,
    label_entity_id: CifField.ofStrings(label_entity_id),
    occupancy: CifField.ofNumbers(mol.occupancy),
    type_symbol: CifField.ofStrings(mol.element),
    pdbx_PDB_ins_code,
    pdbx_PDB_model_num,
    partial_charge: CifField.ofNumbers(mol.charge),
    pdbx_formal_charge: CifField.ofNumbers(mol.formalcharge),
  }
}

async function moleculeToMmCif(
  mol: MoleculeKitDict,
  header: string
): Promise<CifFrame> {
  const entityBuilder = new EntityBuilder()
  const helperCategories: CifCategory[] = []
  const heteroNames: [string, string][] = []

  // TER indices from segment changes
  const terIndices = new Set<number>()
  const segid = mol.segid
  for (let i = 0; i < segid.length - 1; i++) {
    if (segid[i] !== segid[i + 1]) {
      terIndices.add(i + 1)
    }
  }

  const seqIds = Column.ofIntArray(mol.resid)
  const atomIds = Column.ofStringArray(mol.name)
  const compIds = Column.ofStringArray(mol.resname)
  const asymIds = Column.ofStringArray(mol.chain)
  const componentBuilder = new ComponentBuilder(seqIds, atomIds)
  componentBuilder.setNames(heteroNames)
  entityBuilder.setNames(heteroNames)

  const label_entity_id: string[] = []
  for (let i = 0; i < compIds.rowCount; i++) {
    const compId = compIds.value(i)
    const moleculeType = getMoleculeType(
      componentBuilder.add(compId, i).type,
      compId
    )
    label_entity_id[i] = entityBuilder.getEntityId(
      compId,
      moleculeType,
      asymIds.value(i)
    )
  }

  const atom_site = getAtomSite(mol, label_entity_id, terIndices)

  if (mol.bonds.length > 0) {
    helperCategories.push(parseConect(mol, atom_site))
  }

  const categories: Record<string, CifCategory> = {
    entity: CifCategory.ofTable('entity', entityBuilder.getEntityTable()),
    chem_comp: CifCategory.ofTable('chem_comp', componentBuilder.getChemCompTable()),
    atom_site: CifCategory.ofFields('atom_site', atom_site),
  }

  for (const c of helperCategories) {
    categories[c.name] = c
  }

  return {
    header: header || 'MoleculeKit',
    categoryNames: Object.keys(categories),
    categories,
  }
}

function trajectoryProps(trajectory: Trajectory) {
  const first = trajectory.representative
  return {
    label: `${first.entry}`,
    description: `${trajectory.frameCount} model${trajectory.frameCount === 1 ? '' : 's'}`,
  }
}

type TrajectoryFromMoleculeKit = typeof TrajectoryFromMoleculeKit
export const TrajectoryFromMoleculeKit = PluginStateTransform.BuiltIn({
  name: 'trajectory-from-moleculekit',
  display: {
    name: 'Parse MoleculeKit',
    description: 'Parse MoleculeKit dict and create trajectory.',
  },
  from: PluginStateObject.Root,
  to: PluginStateObject.Molecule.Trajectory,
  params: {
    mol: PD.Value<MoleculeKitDict>({} as MoleculeKitDict, { isHidden: true }),
    name: PD.Optional(PD.Text('')),
  },
})({
  apply({ params }) {
    return Task.create('Parse MoleculeKit', async (ctx) => {
      const cif = await moleculeToMmCif(
        params.mol,
        params.name || 'MoleculeKit'
      )
      const format = MmcifFormat.fromFrame(cif, undefined, {
        kind: 'pdb',
        name: 'MoleculeKit',
        data: null,
      } as any)
      const basic = createBasic(format.data.db, true)
      const models = await createModels(basic, format, ctx)

      const numFrames = params.mol.numFrames | 0
      if (numFrames <= 1) {
        return new PluginStateObject.Molecule.Trajectory(models, trajectoryProps(models))
      }

      // Multi-frame: pair the topology Model with a Coordinates object that
      // holds all frames. mol*'s built-in trajectory machinery (frame slider,
      // animations) reads from this.
      const trajectory = buildMultiFrameTrajectory(models.representative, params.mol)
      return new PluginStateObject.Molecule.Trajectory(trajectory, trajectoryProps(trajectory))
    })
  },
})

function buildMultiFrameTrajectory(model: Model, mol: MoleculeKitDict): Trajectory {
  const numAtoms = mol.numAtoms
  const numFrames = mol.numFrames
  const coords = mol.coords
  const frameStride = 3 * numAtoms
  const frames: Frame[] = []
  for (let f = 0; f < numFrames; f++) {
    const off = f * frameStride
    const x = (coords as any).slice(off + 0 * numAtoms, off + 1 * numAtoms) as ArrayLike<number>
    const y = (coords as any).slice(off + 1 * numAtoms, off + 2 * numAtoms) as ArrayLike<number>
    const z = (coords as any).slice(off + 2 * numAtoms, off + 3 * numAtoms) as ArrayLike<number>
    frames.push({
      elementCount: numAtoms,
      time: Time(f, 'step'),
      x, y, z,
      xyzOrdering: { isIdentity: true },
    })
  }
  const coordinates = Coordinates.create(frames, Time(1, 'step'), Time(0, 'step'))
  return Model.trajectoryFromModelAndCoordinates(model, coordinates)
}
