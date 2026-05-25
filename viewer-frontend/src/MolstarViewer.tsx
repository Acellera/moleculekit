import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context'
import { createPluginUI } from 'molstar/lib/mol-plugin-ui'
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18'
import { DefaultPluginUISpec } from 'molstar/lib/mol-plugin-ui/spec'
import { PluginConfig } from 'molstar/lib/mol-plugin/config'
import { setSubtreeVisibility } from 'molstar/lib/mol-plugin/behavior/static/state'
import { StructureElement } from 'molstar/lib/mol-model/structure'
import { OrderedSet } from 'molstar/lib/mol-data/int'
import { PluginStateObject } from 'molstar/lib/mol-plugin-state/objects'
import { StateTransforms } from 'molstar/lib/mol-plugin-state/transforms'
import { TrajectoryFromMoleculeKit } from './moleculeToCIF'
import type { MoleculeKitDict } from './types'
import 'molstar/build/viewer/theme/dark.css'

export type MolstarViewerHandle = {
  hasSlot: (slotId: string) => boolean
  addSlot: (slotId: string, mol: MoleculeKitDict) => Promise<void>
  updateSlotTopology: (slotId: string, mol: MoleculeKitDict) => Promise<void>
  updateSlotCoords: (slotId: string, mol: MoleculeKitDict) => Promise<void>
  removeSlot: (slotId: string) => Promise<void>
  setSlotVisibility: (slotId: string, visible: boolean) => void
}

type SlotRefs = {
  // The TrajectoryFromMoleculeKit transform cell ref (string).
  // Root of the slot's subtree; deleting this deletes everything below.
  trajRef: string
  // Label transform refs created by applyFormalChargeLabelsForSlot.
  labelRefs: string[]
  // Last mol dict applied — used to repaint labels on coord-only updates
  // (labels live at the state root and don't follow the trajectory update).
  lastMol: MoleculeKitDict
}

const MolstarViewer = forwardRef<MolstarViewerHandle, {}>((_props, ref) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const pluginRef = useRef<PluginUIContext | null>(null)
  const slotsRef = useRef<Map<string, SlotRefs>>(new Map())
  // Set once the plugin is ready; awaited by imperative methods invoked
  // before initialization completes (SSE event landing during boot).
  const readyRef = useRef<Promise<void> | null>(null)

  useEffect(() => {
    let cancelled = false
    let resolveReady: () => void
    readyRef.current = new Promise<void>((res) => { resolveReady = res })

    const init = async () => {
      const spec = DefaultPluginUISpec()
      spec.layout = {
        initial: {
          isExpanded: false,
          showControls: false,
          controlsDisplay: 'reactive' as const,
        },
      }
      spec.canvas3d = {
        renderer: {
          backgroundColor: 0x000000 as any,
        },
      }
      spec.components = {
        remoteState: 'none',
        // Hide the developer-oriented left (state tree) and bottom (log)
        // panels. Leave `right` at its default so the user can pop open
        // mol*'s representations/components controls via the wrench icon
        // in the viewport top-right.
        controls: {
          left: 'none',
          bottom: 'none',
        },
      }
      spec.config = [
        [PluginConfig.Viewport.ShowAnimation, true],
        [PluginConfig.Viewport.ShowTrajectoryControls, true],
        [PluginConfig.Viewport.ShowSelectionMode, false],
        [PluginConfig.Viewport.ShowExpand, false],
        // Show the wrench button that toggles the (initially-collapsed)
        // right panel.
        [PluginConfig.Viewport.ShowControls, true],
      ]
      const plugin = await createPluginUI({
        target: containerRef.current!,
        spec,
        render: renderReact18,
      })
      if (cancelled) {
        plugin.dispose()
        return
      }
      pluginRef.current = plugin
      resolveReady()
    }
    init()

    return () => {
      cancelled = true
      pluginRef.current?.dispose()
      pluginRef.current = null
      slotsRef.current.clear()
    }
  }, [])

  const waitReady = async () => {
    if (readyRef.current) await readyRef.current
    return pluginRef.current
  }

  const applyFormalChargeLabelsForSlot = async (slotId: string, mol: MoleculeKitDict) => {
    const plugin = pluginRef.current
    const refs = slotsRef.current.get(slotId)
    if (!plugin || !refs) return

    // Drop previous labels for this slot.
    if (refs.labelRefs.length > 0) {
      const build = plugin.build()
      for (const r of refs.labelRefs) build.delete(r)
      await build.commit()
      refs.labelRefs = []
    }

    const chargedAtoms: { idx: number; q: number }[] = []
    for (let i = 0; i < mol.numAtoms; i++) {
      const q = mol.formalcharge[i]
      if (q) chargedAtoms.push({ idx: i, q })
    }
    if (chargedAtoms.length === 0) return

    // Resolve the slot's structure (one Trajectory → one Model → one Structure
    // when the auto preset has been applied). We scan structures and pick the
    // one whose source trajectory matches our slot's trajRef.
    const structures = plugin.managers.structure.hierarchy.current.structures
    const tree = plugin.state.data.tree
    const inSlotSubtree = new Set<string>()
    const collect = (r: string) => {
      inSlotSubtree.add(r)
      const c = tree.children.get(r)
      if (c) c.forEach((ch: string) => collect(ch))
    }
    collect(refs.trajRef)

    let structureCell: any = null
    for (const s of structures) {
      if (inSlotSubtree.has(s.cell.transform.ref)) {
        structureCell = s.cell
        break
      }
    }
    if (!structureCell) return
    const structure = structureCell.obj?.data
    const structureRef = structureCell.transform.ref
    if (!structure || !structureRef) return

    const atomToUnit = new Map<number, { unit: any; unitIdx: number }>()
    for (const unit of structure.units) {
      if (unit.kind !== 0) continue
      for (let i = 0; i < unit.elements.length; i++) {
        atomToUnit.set(unit.elements[i], { unit, unitIdx: i })
      }
    }

    const build = plugin.state.data.build()
    const pendingRefs: string[] = []
    for (const { idx, q } of chargedAtoms) {
      const hit = atomToUnit.get(idx)
      if (!hit) continue
      const elements = [{ unit: hit.unit, indices: OrderedSet.ofSingleton(hit.unitIdx) }]
      const loci = StructureElement.Loci(structure, elements as any)
      const bundle = StructureElement.Bundle.fromLoci(loci)
      const text = q > 0 ? `+${q}` : `${q}`
      const node = build.toRoot()
        .apply(StateTransforms.Model.MultiStructureSelectionFromBundle, {
          selections: [{ key: `fc-${slotId}-${idx}`, ref: structureRef, groupId: '', bundle }],
          isTransitive: true,
          label: `Formal charge ${text}`,
        } as any, { dependsOn: [structureRef] })
        .apply(StateTransforms.Representation.StructureSelectionsLabel3D, {
          customText: text,
          textColor: 0x000000,
          textSize: 0.4,
          borderColor: 0xffffff,
          borderWidth: 0.25,
          background: false,
          offsetZ: 0.6,
          scaleByRadius: false,
        } as any)
      pendingRefs.push(node.ref)
    }
    await build.commit()
    refs.labelRefs = pendingRefs
  }

  useImperativeHandle(ref, () => ({
    hasSlot: (slotId) => slotsRef.current.has(slotId),

    addSlot: async (slotId, mol) => {
      const plugin = await waitReady()
      if (!plugin) return
      if (slotsRef.current.has(slotId)) {
        // Treat re-adding the same slot as a topology update.
        // (Defensive: caller usually checks hasSlot first.)
        const refs = slotsRef.current.get(slotId)!
        await plugin.build().to(refs.trajRef).update({ mol, name: 'Structure' }).commit()
        refs.lastMol = mol
        await applyFormalChargeLabelsForSlot(slotId, mol)
        return
      }

      const trajectory = await plugin.state.data.build()
        .toRoot()
        .apply(TrajectoryFromMoleculeKit, { mol, name: 'Structure' }, {})
        .commit()
      const trajRef = trajectory.ref

      slotsRef.current.set(slotId, { trajRef, labelRefs: [], lastMol: mol })

      // 'auto' preset for the full hierarchy (cartoon for protein,
      // ball-and-stick for ligands/sugars, etc.).
      await plugin.builders.structure.hierarchy.applyPreset(
        trajectory,
        'default',
        {
          representationPreset: 'auto',
          representationPresetParams: {
            ignoreHydrogens: false,
            theme: { globalName: 'element-symbol', carbonColor: 'chain-id' },
          },
        } as never
      )

      // Drop the carbohydrate (SNFG) reps; bump branched ball-and-stick to
      // full alpha (the auto preset dims sugars expecting SNFG to be primary).
      // Also enable double/aromatic bond rendering on ball-and-stick.
      const repCells = plugin.state.data.selectQ((q: any) =>
        q.ofType(PluginStateObject.Molecule.Structure.Representation3D))
      const build = plugin.state.data.build()
      let dirty = false
      for (const cell of repCells) {
        const params = cell.transform?.params
        if (!params?.type) continue
        if (params.type.name === 'carbohydrate') {
          build.delete(cell.transform.ref)
          dirty = true
        } else if (params.type.params?.alpha != null && params.type.params.alpha < 1) {
          build.to(cell).update({
            ...params,
            type: { ...params.type, params: { ...params.type.params, alpha: 1 } },
          })
          dirty = true
        } else if (params.type.name === 'ball-and-stick') {
          build.to(cell).update({
            ...params,
            type: {
              ...params.type,
              params: { ...params.type.params, multipleBonds: 'symmetric', aromaticBonds: false },
            },
          })
          dirty = true
        }
      }
      if (dirty) await build.commit()

      await applyFormalChargeLabelsForSlot(slotId, mol)
    },

    updateSlotTopology: async (slotId, mol) => {
      const plugin = await waitReady()
      if (!plugin) return
      const refs = slotsRef.current.get(slotId)
      if (!refs) return
      plugin.canvas3d?.setProps({ camera: { manualReset: true } })
      await plugin.build().to(refs.trajRef).update({ mol, name: 'Structure' }).commit()
      refs.lastMol = mol
      await applyFormalChargeLabelsForSlot(slotId, mol)
      setTimeout(() => plugin.canvas3d?.setProps({ camera: { manualReset: false } }), 500)
    },

    updateSlotCoords: async (slotId, mol) => {
      const plugin = await waitReady()
      if (!plugin) return
      const refs = slotsRef.current.get(slotId)
      if (!refs) return
      plugin.canvas3d?.setProps({ camera: { manualReset: true } })
      await plugin.build().to(refs.trajRef).update({ mol, name: 'Structure' }).commit()
      refs.lastMol = mol
      setTimeout(() => plugin.canvas3d?.setProps({ camera: { manualReset: false } }), 500)
    },

    removeSlot: async (slotId) => {
      const plugin = await waitReady()
      const refs = slotsRef.current.get(slotId)
      if (!plugin || !refs) return
      const build = plugin.build()
      for (const r of refs.labelRefs) build.delete(r)
      build.delete(refs.trajRef)
      await build.commit()
      slotsRef.current.delete(slotId)
    },

    setSlotVisibility: (slotId, visible) => {
      const plugin = pluginRef.current
      const refs = slotsRef.current.get(slotId)
      if (!plugin || !refs) return
      // setSubtreeVisibility walks the cell tree and toggles visibility on
      // every downstream representation, not just the root cell.
      setSubtreeVisibility(plugin.state.data, refs.trajRef, !visible)
      for (const r of refs.labelRefs) {
        setSubtreeVisibility(plugin.state.data, r, !visible)
      }
    },
  }))

  return (
    <div
      ref={containerRef}
      style={{ position: 'absolute', inset: 0, background: '#101015' }}
    />
  )
})

MolstarViewer.displayName = 'MolstarViewer'

export default MolstarViewer
