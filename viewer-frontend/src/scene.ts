/**
 * Apply a moleculekit scene description to a Mol* plugin.
 *
 * The description is decided in Python (moleculekit/viewer/molstar/scene.py) so
 * that the interactive viewer and the headless renderer cannot drift: this file
 * is the single place that turns one into Mol* state, and both bundle entries
 * import it.
 */
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder'
import { StateTransforms } from 'molstar/lib/mol-plugin-state/transforms'
import { createStructureRepresentationParams } from 'molstar/lib/mol-plugin-state/helpers/structure-representation-params'
import { StructureSelectionQueries } from 'molstar/lib/mol-plugin-state/helpers/structure-selection-query'
import { StructureQueryHelper } from 'molstar/lib/mol-plugin-state/helpers/structure-query'
import { Structure, StructureSelection } from 'molstar/lib/mol-model/structure'
import { Loci } from 'molstar/lib/mol-model/loci'
import { Vec3 } from 'molstar/lib/mol-math/linear-algebra'
import { Color } from 'molstar/lib/mol-util/color'
import { ColorNames } from 'molstar/lib/mol-util/color/names'
import type { PluginContext } from 'molstar/lib/mol-plugin/context'
import type { StateBuilder, StateObjectSelector } from 'molstar/lib/mol-state'

export interface SceneSelect {
  kind: 'builtin' | 'resname' | 'atoms'
  name?: string
  names?: string[]
  indices?: number[]
}

export interface SceneComponent {
  select: SceneSelect
  representation: { type: string; size_factor?: number }
  color?: { theme?: string; uniform?: string }
  opacity?: number
}

export interface SceneLabel {
  atom: number
  position: number[]
  text: string
  size: number
  color: string
  offset: number
}

export interface Scene {
  components: SceneComponent[]
  labels?: SceneLabel[]
  tubes?: { start: number[]; end: number[]; radius: number; color: string }[]
  camera?: { direction?: number[]; up?: number[]; radius_factor?: number; focus?: SceneSelect }
  canvas?: { background?: string }
}

/** Translate an SVG colour name or hex string into a Mol* packed colour. */
function colorOf(value: string): Color {
  const named = (ColorNames as Record<string, Color>)[value.toLowerCase()]
  if (named !== undefined) return named
  return Color(parseInt(value.replace('#', ''), 16))
}

// These mirror MVS's own resolution of its static selectors, not the bare
// StructureSelectionQueries entries of the same name: MVS's static
// component type ('ligand', 'branched', ...) is resolved by
// createStructureComponent in
// mol-plugin-state/helpers/structure-component.js, which maps 'ligand' to
// StructureSelectionQueries.ligandPlusConnected (the ligand plus one layer
// of covalently/metal-connected whole residues, minus anything branched)
// and 'branched' to StructureSelectionQueries.branchedPlusConnected, not to
// the plain .ligand/.branched queries. Using the plain queries here would
// select fewer atoms than MVS does for a covalently bound ligand or a
// glycan, which is exactly the drift this shared description exists to
// remove. 'polymer', 'ion' and 'all' do resolve to their plain queries in
// that same file, so those three are used as-is.
const BUILTIN: Record<string, any> = {
  polymer: StructureSelectionQueries.polymer.expression,
  ligand: StructureSelectionQueries.ligandPlusConnected.expression,
  ion: StructureSelectionQueries.ion.expression,
  water: StructureSelectionQueries.water.expression,
  branched: StructureSelectionQueries.branchedPlusConnected.expression,
}

// MVS maps its atom_index onto Mol*'s sourceIndex, the atom's position in the
// input file, which is exactly the moleculekit atom index our description uses.
function expression(select: SceneSelect): any {
  if (select.kind === 'builtin') {
    if (select.name === 'all') return MS.struct.generator.all()
    const query = BUILTIN[select.name!]
    if (!query) throw new Error(`unknown builtin selector: ${select.name}`)
    return query
  }
  if (select.kind === 'resname') {
    return MS.struct.generator.atomGroups({
      'residue-test': MS.core.set.has([
        // MS.set(...xs) itself just collects xs into an array and hands it to
        // core.type.set; calling that directly skips spreading the array as
        // call arguments, which a large atom-index selection could overflow.
        MS.core.type.set(select.names ?? []),
        MS.ammp('label_comp_id'),
      ]),
    })
  }
  return MS.struct.generator.atomGroups({
    'atom-test': MS.core.set.has([
      MS.core.type.set(select.indices ?? []),
      MS.acp('sourceIndex'),
    ]),
  })
}

/**
 * Resolve a scene colour into `createStructureRepresentationParams` props.
 *
 * A uniform colour must land on the `color`/`colorParams` keys (the uniform
 * colour *theme*), not on `type`: `type` is the representation itself
 * (cartoon, ball-and-stick, ...), and overwriting it with a theme name there
 * makes Mol* try to use "uniform" as a representation type.
 */
function themeParams(color?: { theme?: string; uniform?: string }) {
  if (color?.uniform) {
    return { color: 'uniform', colorParams: { value: colorOf(color.uniform) } }
  }
  return { color: color?.theme ?? 'element-symbol' }
}

/**
 * Translate one MVS representation type name into Mol*'s native
 * representation type, type params and size theme.
 *
 * MVS spells representation types in its own vocabulary (snake_case names,
 * a `size_factor` field) which is not what Mol*'s own representation
 * registry uses (hyphenated names, and a `sizeFactor` type param that means
 * something else there: the sphere/cylinder radius ratio, not an overall
 * scale). Passing an MVS name straight through does not error: Mol*'s
 * `PD.Mapped` silently resolves an unrecognised name to the first
 * applicable representation for the structure, so a `ball_and_stick`
 * component drawn on a ligand-only selection quietly became an invisible
 * cartoon. This mirrors MVS's own translation
 * (`extensions/mvs/load-helpers.js`'s `representationPropsBase`) for every
 * type `build_scene()` can currently emit, so applyScene draws the same
 * picture MVS does for the same description. `size_factor` defaults to 1,
 * MVS's own default for an omitted field (`mvs-tree-representations.js`).
 */
function representationParams(representation: SceneComponent['representation']): {
  type: string
  typeParams: Record<string, unknown>
  size: string
  sizeParams: Record<string, unknown>
} {
  const value = representation.size_factor ?? 1
  switch (representation.type) {
    case 'cartoon':
      return { type: 'cartoon', typeParams: {}, size: 'uniform', sizeParams: { value } }
    case 'ball_and_stick':
      return {
        type: 'ball-and-stick',
        typeParams: { sizeFactor: 0.5, sizeAspectRatio: 0.5 },
        size: 'uniform',
        sizeParams: { value },
      }
    case 'line':
      return { type: 'line', typeParams: {}, size: 'uniform', sizeParams: { value } }
    case 'spacefill':
      return { type: 'spacefill', typeParams: {}, size: 'physical', sizeParams: { scale: value } }
    default:
      throw new Error(`applyScene cannot render representation type: ${representation.type}`)
  }
}

/** Build one component's representation and add it to `build`. */
function applyComponent(
  plugin: PluginContext,
  build: StateBuilder.Root,
  structure: StateObjectSelector,
  comp: SceneComponent
): void {
  const selection = build
    .to(structure)
    .apply(StateTransforms.Model.StructureSelectionFromExpression, {
      expression: expression(comp.select),
    })
  const repr = representationParams(comp.representation)
  const params: any = createStructureRepresentationParams(
    plugin,
    undefined,
    {
      type: repr.type as any,
      typeParams: repr.typeParams,
      size: repr.size as any,
      sizeParams: repr.sizeParams,
      ...themeParams(comp.color),
    } as any
  )
  if (comp.opacity !== undefined) {
    params.type.params.alpha = comp.opacity
  }
  selection.apply(StateTransforms.Representation.StructureRepresentation3D, params)
}

/** Label one atom, reusing the transform the interactive viewer uses for the
 * same purpose (see MolstarViewer.tsx's formal-charge labels). A label
 * attaches to a *selection*, not a bare coordinate, so this builds a
 * single-atom selection from the label's atom index first. */
function applyLabel(
  build: StateBuilder.Root,
  structure: StateObjectSelector,
  label: SceneLabel
): void {
  build
    .toRoot()
    .apply(
      StateTransforms.Model.MultiStructureSelectionFromExpression,
      {
        selections: [
          {
            key: `label-${label.atom}`,
            ref: structure.ref,
            groupId: '',
            expression: expression({ kind: 'atoms', indices: [label.atom] }),
          },
        ],
        isTransitive: true,
        label: label.text,
      },
      { dependsOn: [structure.ref] }
    )
    .apply(StateTransforms.Representation.StructureSelectionsLabel3D, {
      customText: label.text,
      textColor: colorOf(label.color),
      textSize: label.size,
      borderColor: colorOf('white'),
      borderWidth: 0.25,
      background: false,
      offsetZ: label.offset,
      scaleByRadius: false,
    } as any)
}

/** Run a selector against the realised structure and return its element loci.
 *
 * This is the same primitive `MultiStructureSelectionFromExpression` uses
 * internally (compile the expression, run it, convert the selection to a
 * loci), just invoked directly instead of through plugin state, since the
 * camera needs a bounding sphere rather than a new state object. */
function lociFor(structure: StateObjectSelector, select: SceneSelect) {
  const { selection } = StructureQueryHelper.createAndRun(
    structure.data as Structure,
    expression(select)
  )
  return StructureSelection.toLociWithSourceUnits(selection)
}

/** Move the camera onto `camera`'s focus, direction and radius. */
async function applyCamera(
  plugin: PluginContext,
  structure: StateObjectSelector,
  camera: NonNullable<Scene['camera']>
): Promise<void> {
  const canvas3d = plugin.canvas3d!
  const loci = camera.focus
    ? lociFor(structure, camera.focus)
    : Structure.toStructureElementLoci(structure.data as Structure)
  const sphere = Loci.getBoundingSphere(loci)!
  const dir = camera.direction
    ? Vec3.create(camera.direction[0], camera.direction[1], camera.direction[2])
    : undefined
  const up = camera.up ? Vec3.create(camera.up[0], camera.up[1], camera.up[2]) : undefined
  const radius = sphere.radius * (camera.radius_factor ?? 1)
  const snapshot = canvas3d.camera.getFocus(sphere.center, radius)
  // getFocus honours `dir` and `up` only up to sign: it runs them through
  // Vec3.matchDirection, which flips them into the hemisphere the camera
  // already looks along. Front and back, left and right, top and bottom then
  // all resolve to one image. Place the camera from the vectors directly.
  // `direction` points from the camera toward the target, so the position is
  // that far back along it.
  if (dir) {
    const distance = Vec3.distance(snapshot.position!, snapshot.target!)
    const offset = Vec3.setMagnitude(Vec3.zero(), dir, distance)
    snapshot.position = Vec3.sub(Vec3.zero(), snapshot.target!, offset)
  }
  if (up) snapshot.up = Vec3.clone(up)
  // Go through the pending-reset mechanism instead of camera.setState
  // directly: committing new representations (just above, in applyScene)
  // schedules Mol*'s own automatic camera-fit-to-scene reset
  // (canvas3d.js's resolveCameraReset), which runs later in the animation
  // loop and recomputes target/radius from the whole visible scene,
  // overwriting a direct setState call outright. requestCameraReset queues
  // our snapshot as `nextCameraResetSnapshot`, which resolveCameraReset
  // merges *over* its own auto-fit result instead of racing it -- the same
  // mechanism Mol*'s CameraManager.focusSphere (and MVS's own camera
  // handling) uses.
  // durationMs 0: an animated transition would let a screenshot land
  // mid-interpolation, which is how repeated renders stopped being identical
  // once before.
  canvas3d.requestCameraReset({ snapshot, durationMs: 0 })
}

/**
 * Apply a scene description to a Mol* plugin for one already-loaded structure.
 *
 * Builds every component's representation, adds formal-charge-style labels,
 * sets the canvas background and moves the camera, in that order. `tubes` is
 * not supported here: neither `render()` nor the interactive viewer ever
 * populates it, since it only exists for the MolViewSpec-only docs/notebook
 * path (see mvs.py), so a non-empty `tubes` list is rejected rather than
 * silently dropped.
 *
 * @param plugin the plugin whose state is built into
 * @param structure a Structure state selector for the molecule being described
 * @param scene the description produced by build_scene() in Python
 */
export async function applyScene(
  plugin: PluginContext,
  structure: StateObjectSelector,
  scene: Scene
): Promise<void> {
  if (scene.tubes?.length) {
    throw new Error('applyScene cannot render tubes; highlight_bonds is MVS-only')
  }

  const build = plugin.state.data.build()
  for (const comp of scene.components) {
    applyComponent(plugin, build, structure, comp)
  }
  for (const label of scene.labels ?? []) {
    applyLabel(build, structure, label)
  }
  await build.commit()

  if (scene.canvas?.background && plugin.canvas3d) {
    const hex = scene.canvas.background
    plugin.canvas3d.setProps({ renderer: { backgroundColor: colorOf(hex) } })
  }

  if (scene.camera) {
    await applyCamera(plugin, structure, scene.camera)
  }
}
