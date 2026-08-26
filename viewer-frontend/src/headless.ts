/**
 * Headless render entry point. Driven from Python over CDP by
 * moleculekit/viewer/molstar/render.py. No UI, no user input: a plugin, a
 * canvas, an MVS scene and an ImagePass screenshot.
 */
import { PluginContext } from 'molstar/lib/mol-plugin/context'
import { DefaultPluginSpec, PluginSpec } from 'molstar/lib/mol-plugin/spec'
import { loadMVS } from 'molstar/lib/extensions/mvs/load'
import { MVSData } from 'molstar/lib/extensions/mvs/mvs-data'
import { MolViewSpec } from 'molstar/lib/extensions/mvs/behavior'
import { ParamDefinition as PD } from 'molstar/lib/mol-util/param-definition'
import { SsaoParams } from 'molstar/lib/mol-canvas3d/passes/ssao'

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

  // loadMVS refuses to run unless the MolViewSpec behavior is registered:
  // DefaultPluginSpec() alone does not include it.
  const spec = DefaultPluginSpec()
  spec.behaviors.push(PluginSpec.Behavior(MolViewSpec))
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


async function load(mvsj: string): Promise<void> {
  if (!plugin) throw new Error('init() must run before load()')
  const canvas3d = plugin.canvas3d!
  // didDraw is a BehaviorSubject: subscribing to it always replays whatever
  // value it last held, even if that draw predates this load. Capture the
  // baseline before loading and wait for a strictly later draw, so we do
  // not resolve on a stale replay of a draw from init() or a prior load().
  const baseline = canvas3d.didDraw.value
  // appendSnapshots defaults to false, which already replaces the existing
  // scene; spelled out here because that is exactly the behavior we need.
  await loadMVS(plugin, MVSData.fromMVSJ(mvsj), { appendSnapshots: false })
  // Resolve only once something has actually been drawn after this load.
  // Without this the screenshot can be taken against an empty scene.
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



;(window as any).mkHeadless = { init, load, screenshot, glInfo }
