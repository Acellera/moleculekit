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
import type { StateObjectSelector } from 'molstar/lib/mol-state'
import { applyScene, type Scene } from './scene'

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

async function applySceneAndDraw(scene: Scene): Promise<void> {
  if (!plugin || !structureRef) throw new Error('loadStructure() must run first')
  const canvas3d = plugin.canvas3d!
  // didDraw is a BehaviorSubject: subscribing to it always replays whatever
  // value it last held, even if that draw predates this call. Capture the
  // baseline before applying and wait for a strictly later draw, so we do
  // not resolve on a stale replay of a draw from init() or a prior scene.
  const baseline = canvas3d.didDraw.value
  await applyScene(plugin, structureRef, scene)
  // Resolve only once something has actually been drawn after this. Without
  // this the screenshot can be taken against an empty or half-built scene.
  await new Promise<void>((resolve) => {
    const sub = canvas3d.didDraw.subscribe((t) => {
      if (t > baseline) {
        sub.unsubscribe()
        resolve()
      }
    })
  })
}


async function screenshot(opts: {
  width: number
  height: number
  occlusion: boolean
  transparent: boolean
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
  return await helper.getImageDataUri()
}



;(window as any).mkHeadless = {
  init,
  setViewport,
  loadStructure,
  applyScene: applySceneAndDraw,
  screenshot,
  glInfo,
}
