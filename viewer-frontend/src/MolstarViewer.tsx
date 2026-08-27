import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context'
import { createPluginUI } from 'molstar/lib/mol-plugin-ui'
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18'
import { DefaultPluginUISpec } from 'molstar/lib/mol-plugin-ui/spec'
import { PluginConfig } from 'molstar/lib/mol-plugin/config'
import { setSubtreeVisibility } from 'molstar/lib/mol-plugin/behavior/static/state'
import { StateTransforms } from 'molstar/lib/mol-plugin-state/transforms'
import { TrajectoryFromMoleculeKit } from './moleculeToCIF'
import { applyScene, type Scene } from './scene'
import type { MoleculeKitDict } from './types'
import 'molstar/build/viewer/theme/light.css'

export type MolstarViewerHandle = {
  hasSlot: (slotId: string) => boolean
  addSlot: (slotId: string, mol: MoleculeKitDict, scene: Scene) => Promise<void>
  updateSlotTopology: (slotId: string, mol: MoleculeKitDict, scene: Scene) => Promise<void>
  updateSlotCoords: (slotId: string, mol: MoleculeKitDict) => Promise<void>
  removeSlot: (slotId: string) => Promise<void>
  setSlotVisibility: (slotId: string, visible: boolean) => void
}

type SlotRefs = {
  // The TrajectoryFromMoleculeKit transform cell ref (string).
  // Root of the slot's subtree; deleting this deletes everything below.
  trajRef: string
  // Label transform refs applyScene created for this slot (see
  // collectLabelRefs), so removeSlot/setSlotVisibility can find them.
  labelRefs: string[]
  // Last mol dict applied. Currently unused: written in four places below,
  // read nowhere.
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
          backgroundColor: 0xffffff as any,
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

  // applyScene (scene.ts) creates formal-charge-style labels rooted at the
  // state root (see its applyLabel), not under the slot's trajectory
  // subtree, so deleting the trajectory does not delete them. Find the refs
  // it just created for `structureRef` so removeSlot and setSlotVisibility
  // can still find and clean up the label nodes.
  const collectLabelRefs = (structureRef: string): string[] => {
    const plugin = pluginRef.current
    if (!plugin) return []
    const tree = plugin.state.data.tree
    const rootChildren = tree.children.get(tree.root.ref)
    const found: string[] = []
    rootChildren?.forEach((r: string) => {
      const transform = plugin.state.data.cells.get(r)?.transform
      if (
        transform?.transformer === StateTransforms.Model.MultiStructureSelectionFromExpression &&
        transform.dependsOn?.includes(structureRef)
      ) {
        found.push(r)
      }
    })
    return found
  }

  // Delete whatever structure/representation/label subtree currently hangs
  // off `refs.trajRef` (none, the first time a slot is built) and rebuild it
  // fresh from `scene`. The set of components and their colours/types are
  // fixed params baked into each representation node when applyScene runs;
  // recomputing an existing node against a new structure (a plain trajectory
  // update, e.g. on a topology change) keeps those params unchanged, so a
  // rep edit such as a different colour would never show up on a later
  // view() call without rebuilding the nodes themselves. `refs.trajRef`
  // itself is left alone: updateSlotCoords and the trajectory/animation
  // controls depend on that ref staying stable.
  const rebuildSceneForSlot = async (
    plugin: PluginUIContext,
    refs: SlotRefs,
    scene: Scene
  ): Promise<void> => {
    const teardown = plugin.build()
    for (const r of refs.labelRefs) teardown.delete(r)
    plugin.state.data.tree.children.get(refs.trajRef)?.forEach((child: string) => {
      teardown.delete(child)
    })
    await teardown.commit()

    const structure = await plugin.builders.structure.createStructure(
      await plugin.builders.structure.createModel(refs.trajRef)
    )
    await applyScene(plugin, structure, scene)
    refs.labelRefs = collectLabelRefs(structure.ref)
  }

  useImperativeHandle(ref, () => ({
    hasSlot: (slotId) => slotsRef.current.has(slotId),

    addSlot: async (slotId, mol, scene) => {
      const plugin = await waitReady()
      if (!plugin) return
      if (slotsRef.current.has(slotId)) {
        // Treat re-adding the same slot as a topology update.
        // (Defensive: caller usually checks hasSlot first.)
        const refs = slotsRef.current.get(slotId)!
        await plugin.build().to(refs.trajRef).update({ mol, name: 'Structure' }).commit()
        refs.lastMol = mol
        await rebuildSceneForSlot(plugin, refs, scene)
        return
      }

      const trajectory = await plugin.state.data.build()
        .toRoot()
        .apply(TrajectoryFromMoleculeKit, { mol, name: 'Structure' }, {})
        .commit()
      const trajRef = trajectory.ref

      const refs: SlotRefs = { trajRef, labelRefs: [], lastMol: mol }
      slotsRef.current.set(slotId, refs)

      await rebuildSceneForSlot(plugin, refs, scene)
    },

    updateSlotTopology: async (slotId, mol, scene) => {
      const plugin = await waitReady()
      if (!plugin) return
      const refs = slotsRef.current.get(slotId)
      if (!refs) return
      plugin.canvas3d?.setProps({ camera: { manualReset: true } })
      await plugin.build().to(refs.trajRef).update({ mol, name: 'Structure' }).commit()
      refs.lastMol = mol
      await rebuildSceneForSlot(plugin, refs, scene)
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
      style={{ position: 'absolute', inset: 0, background: '#ffffff' }}
    />
  )
})

MolstarViewer.displayName = 'MolstarViewer'

export default MolstarViewer
