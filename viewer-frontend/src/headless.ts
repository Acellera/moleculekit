/**
 * Headless render entry point. Driven from Python over CDP by
 * moleculekit/viewer/molstar/render.py. No UI, no user input: a plugin, a
 * canvas, a scene applied through applyScene (see scene.ts) and an
 * ImagePass screenshot.
 */
import { PluginContext } from 'molstar/lib/mol-plugin/context'
import { DefaultPluginSpec } from 'molstar/lib/mol-plugin/spec'
import { ParamDefinition as PD } from 'molstar/lib/mol-util/param-definition'
import { SsaoParams } from 'molstar/lib/mol-canvas3d/passes/ssao'
import type { Canvas3D } from 'molstar/lib/mol-canvas3d/canvas3d'
import type { StateObjectSelector } from 'molstar/lib/mol-state'
import { applyScene, applyVolume, type Scene, type VolumeRep } from './scene'

let plugin: PluginContext | null = null

function glInfo(): string {
  try {
    const probe = document.createElement('canvas')
    const gl = (probe.getContext('webgl2') ||
      probe.getContext('webgl')) as WebGLRenderingContext | null
    if (!gl) return 'NO WEBGL CONTEXT'
    const dbg = gl.getExtension('WEBGL_debug_renderer_info')
    if (!dbg) return 'WEBGL (renderer name unavailable)'
    return String(gl.getParameter((dbg as any).UNMASKED_RENDERER_WEBGL))
  } catch (e) {
    return `WEBGL PROBE THREW: ${e}`
  }
}

async function init(width: number, height: number): Promise<string> {
  const container = document.getElementById('app') as HTMLDivElement
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  container.appendChild(canvas)

  const spec = DefaultPluginSpec()
  plugin = new PluginContext(spec)
  await plugin.init()
  // initViewer is async in this Mol* version: it wraps the private,
  // formerly-public synchronous initializer.
  const ok = await plugin.initViewerAsync(canvas, container)
  if (!ok) {
    throw new Error(`Mol* could not initialise a WebGL viewer. GL reports: ${glInfo()}`)
  }
  // Every render states its camera (scene.ts applyCamera), so Mol*'s own reset
  // only fights it. That reset fires when a commit adds geometry outside the
  // old bounds, which a scene of several structures does once per structure,
  // and it animates to a fit of the whole scene for 250 ms. A camera requested
  // during the move only retargets it, so the screenshot caught the camera in
  // flight: three renders of one scene gave three framings, and on a render
  // server sometimes a blank image and blank images after it.
  // Orthographic, as in pmview: an atom has the same size at all depths, so
  // atoms near the camera do not grow and cover the subject.
  plugin.canvas3d!.setProps({ camera: { manualReset: true, mode: 'orthographic' } })
  await warmScreenshotPass()
  return glInfo()
}

/**
 * Build the offscreen capture pass here, on a throwaway image.
 *
 * ViewportScreenshotHelper creates its ImagePass on first access and only
 * re-applies props on every access after that, so the first capture in a
 * process runs a different path from all the later ones and comes out with
 * different pixels. Spending that first capture on an empty 64x64 image means
 * every real render takes the same path, so two renders of one scene are
 * byte-identical rather than merely similar.
 */
async function warmScreenshotPass(): Promise<void> {
  const helper = (plugin as any).helpers.viewportScreenshot
  helper.behaviors.values.next({
    ...helper.values,
    resolution: { name: 'custom', params: { width: 64, height: 64 } },
    transparent: false,
    axes: { name: 'off', params: {} },
  })
  await helper.getImageDataUri()
}


/**
 * Size the canvas to the image about to be rendered.
 *
 * The camera's fit distance comes from the canvas viewport, so without this it
 * is computed once against whatever the first render made the canvas and never
 * changes: every later size only cropped or padded that framing. A 1200x900
 * and a 900x900 request both drew the structure 516x473, and which framing you
 * got depended on the first render in the process.
 *
 * Only the aspect actually reaches the image (ImagePass renders the real
 * resolution offscreen, and 2400x1800 is byte-identical whether the canvas is
 * 2400x1800 or 1009x757), but matching the size outright gets the aspect exact
 * where scaling to a fixed height rounds it: 757 * 600/900 is 379, an aspect
 * of 0.5007 rather than 0.5, and those pixels differ.
 */
async function setViewport(width: number, height: number): Promise<void> {
  if (!plugin) throw new Error('init() must run before setViewport()')
  const container = document.getElementById('app') as HTMLDivElement
  container.style.height = `${height}px`
  container.style.width = `${width}px`
  // plugin.handleResize, not canvas3d.handleResize: only the plugin's resizes
  // the canvas from the container (resizeCanvas reads container.offsetWidth).
  // canvas3d's just re-reads the drawing buffer, so on its own the buffer
  // stays whatever init() set it to for the first render, forever.
  plugin.handleResize()
}

let structureRef: StateObjectSelector | null = null
let structureRefs: StateObjectSelector[] = []

/** Parse a base64 BinaryCIF structure and make it the current one. */
async function loadStructure(bcifBase64: string): Promise<void> {
  if (!plugin) throw new Error('init() must run before loadStructure()')
  await plugin.clear()
  const bytes = Uint8Array.from(atob(bcifBase64), (c) => c.charCodeAt(0))
  const data = await plugin.builders.data.rawData({ data: bytes })
  // Mol*'s trajectory format registry keys BinaryCIF under 'mmcif', which
  // parses binary CIF transparently; the same quirk moleculeToCIF.ts's
  // DCD path documents.
  const trajectory = await plugin.builders.structure.parseTrajectory(data, 'mmcif')
  const model = await plugin.builders.structure.createModel(trajectory)
  structureRef = await plugin.builders.structure.createStructure(model)
}

/**
 * Parse several structures into one scene, replacing whatever was there.
 *
 * A figure often shows objects that were loaded separately, a protein and a
 * docked ligand say, and merging them into one structure first would lose
 * which atoms belong to which. Each keeps its own representations instead.
 */
async function loadStructures(bcifBase64List: string[]): Promise<void> {
  if (!plugin) throw new Error('init() must run before loadStructures()')
  await plugin.clear()
  structureRefs = []
  for (const bcifBase64 of bcifBase64List) {
    const bytes = Uint8Array.from(atob(bcifBase64), (c) => c.charCodeAt(0))
    const data = await plugin.builders.data.rawData({ data: bytes })
    const trajectory = await plugin.builders.structure.parseTrajectory(data, 'mmcif')
    const model = await plugin.builders.structure.createModel(trajectory)
    structureRefs.push(await plugin.builders.structure.createStructure(model))
  }
  structureRef = structureRefs[0] ?? null
}

/** Decode base64 to bytes. */
function base64Bytes(b64: string): Uint8Array<ArrayBuffer> {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

/** Inflate a gzip member, using the platform's own decompressor. */
async function gunzip(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

/**
 * Add volumes to the scene the structures were loaded into.
 *
 * Sent as a gzipped CCP4/MRC map rather than cube text: a 200-cell grid is
 * eight million values, which as ASCII is a payload of a hundred megabytes
 * and as float32 a thirtieth of that.
 */
async function loadVolumes(volumes: { ccp4_gz: string; reps: VolumeRep[] }[]): Promise<void> {
  if (!plugin) throw new Error('init() must run before loadVolumes()')
  for (const volume of volumes) {
    const bytes = await gunzip(base64Bytes(volume.ccp4_gz))
    const data = await plugin.builders.data.rawData({ data: bytes })
    const parsed = await plugin.dataFormats.get('ccp4')!.parse(plugin, data)
    const ref = parsed.volumes?.[0] ?? parsed.volume
    await applyVolume(plugin, ref, volume.reps)
  }
}

/**
 * Commit what the scenes built, camera included, and wait for a draw of it.
 *
 * Without this the screenshot can be taken against an empty or half-built
 * scene. The commit is synchronous because several structures otherwise
 * commit over several frames, and the camera request resolves only after the
 * last of them: a draw counted from before the scenes was often one with the
 * previous render's camera.
 */
async function settle(canvas3d: Canvas3D): Promise<void> {
  canvas3d.commit(true)
  // didDraw is a BehaviorSubject: subscribing to it always replays whatever
  // value it last held, so wait for a strictly later draw.
  const baseline = canvas3d.didDraw.value
  canvas3d.requestDraw()
  await new Promise<void>((resolve) => {
    const sub = canvas3d.didDraw.subscribe((t) => {
      if (t > baseline) {
        sub.unsubscribe()
        resolve()
      }
    })
  })
}

/**
 * Apply one scene per structure, then the parts that belong to the whole
 * picture: the background, and a camera that frames every object at once.
 */
async function applyScenesAndDraw(scenes: Scene[], globals: Scene): Promise<void> {
  if (!plugin) throw new Error('loadStructures() must run first')
  for (let i = 0; i < scenes.length; i++) {
    await applyScene(plugin, structureRefs[i], scenes[i])
  }
  // The camera and canvas belong to the picture, not to any one object, so
  // they are applied once at the end against the first structure.
  await applyScene(plugin, structureRefs[0], globals)
  await settle(plugin.canvas3d!)
}

async function applySceneAndDraw(scene: Scene): Promise<void> {
  if (!plugin || !structureRef) throw new Error('loadStructure() must run first')
  await applyScene(plugin, structureRef, scene)
  await settle(plugin.canvas3d!)
}


async function screenshot(opts: {
  width: number
  height: number
  occlusion: boolean
  transparent: boolean
  sampleLevel: number
}): Promise<string> {
  if (!plugin) throw new Error('init() must run before screenshot()')
  // The 'on' variant's params is SsaoParams (samples, multiScale, radius,
  // bias, blurKernelSize, ...): setProps merges postprocessing with a
  // shallow Object.assign, so an incomplete params object here is not
  // filled in with defaults, it is used as-is and later crashes
  // PostprocessingPass.updateState on the next render, permanently killing
  // the render loop (and every load() after it, since load() awaits a
  // draw that will never come). Always pass a complete params object.
  plugin.canvas3d!.setProps({
    postprocessing: {
      occlusion: opts.occlusion
        ? { name: 'on', params: PD.getDefaultValues(SsaoParams) }
        : { name: 'off', params: {} },
    },
  })
  const helper = (plugin as any).helpers.viewportScreenshot
  helper.behaviors.values.next({
    ...helper.values,
    resolution: {
      name: 'custom',
      params: { width: opts.width, height: opts.height },
    },
    transparent: opts.transparent,
    axes: { name: 'off', params: {} },
  })
  // The screenshot helper hardcodes sampleLevel 4, which is 16 full renders of
  // the scene accumulated for anti-aliasing. That is invisible on a GPU and
  // ruinous on the software rasteriser, and it is why a render took far longer
  // than the interactive viewer feels: the viewer multi-samples temporally,
  // one pass per frame, refining only once it goes idle. The helper re-applies
  // only a few props on each access and multiSample is not among them, so it
  // is set on the pass itself, after the getter has run.
  helper.imagePass.setProps({
    multiSample: { mode: 'on', sampleLevel: opts.sampleLevel, reuseOcclusion: false },
  })
  return await helper.getImageDataUri()
}



;(window as any).mkHeadless = {
  init,
  setViewport,
  loadStructure,
  loadStructures,
  loadVolumes,
  applyScene: applySceneAndDraw,
  applyScenes: applyScenesAndDraw,
  screenshot,
  glInfo,
}
