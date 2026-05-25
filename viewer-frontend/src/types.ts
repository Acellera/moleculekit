// Mirrors moleculekit/viewer/molstar/serialize.py output.
export type MoleculeKitDict = {
  altloc: string[]
  atomtype: string[]
  beta: number[]
  bonds: number[][]
  bondtype: string[]
  chain: string[]
  charge: number[]
  element: string[]
  formalcharge: number[]
  insertion: string[]
  name: string[]
  occupancy: number[]
  record: string[]
  resid: number[]
  resname: string[]
  segid: string[]
  serial: number[]
  // Per-frame layout: [x0..xN, y0..yN, z0..zN]. Multi-frame trajectories
  // concatenate frames into one flat buffer.
  coords: number[] | Float32Array
  numFrames: number
  numAtoms: number
}

export type SSETopology = {
  type: 'topology'
  slot: string
  label: string
  mol: MoleculeKitDict
  coords_url: string
  numFrames: number
}

export type SSECoords = {
  type: 'coords'
  slot: string
  coords_url: string
  numFrames: number
}

export type SSERemove = { type: 'remove'; slot: string }

export type SSEEvent = SSETopology | SSECoords | SSERemove

export type SlotState = {
  uuid: string
  label: string
  visible: boolean
}
