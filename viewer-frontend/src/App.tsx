import { useEffect, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react'
import MolstarViewer, { type MolstarViewerHandle } from './MolstarViewer'
import { connect, type SSEStatus } from './sse'
import type { SSEEvent, MoleculeKitDict, SlotState } from './types'

function readSession(): string {
  const params = new URLSearchParams(window.location.search)
  return params.get('session') ?? ''
}

async function fetchCoords(url: string): Promise<Float32Array> {
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`coords fetch failed: ${resp.status}`)
  const buf = await resp.arrayBuffer()
  return new Float32Array(buf)
}

// Caches per-slot mol topology so coords-only events can reuse it.
const molByUuid = new Map<string, MoleculeKitDict>()

export default function App() {
  const [slots, setSlots] = useState<SlotState[]>([])
  const [status, setStatus] = useState<SSEStatus>({ kind: 'open' })
  const viewerRef = useRef<MolstarViewerHandle>(null)

  useEffect(() => {
    const session = readSession()
    const close = connect(
      session,
      async (raw) => {
        const ev = raw as SSEEvent
        const v = viewerRef.current
        if (!v) return

        if (ev.type === 'topology') {
          const allCoords = await fetchCoords(ev.coords_url)
          const mol: MoleculeKitDict = { ...ev.mol, coords: allCoords, numFrames: ev.numFrames }
          molByUuid.set(ev.slot, mol)
          if (v.hasSlot(ev.slot)) {
            await v.updateSlotTopology(ev.slot, mol)
          } else {
            await v.addSlot(ev.slot, mol)
          }
          setSlots((prev) => {
            if (prev.some((s) => s.uuid === ev.slot)) {
              return prev.map((s) => (s.uuid === ev.slot ? { ...s, label: ev.label } : s))
            }
            return [...prev, { uuid: ev.slot, label: ev.label, visible: true }]
          })
        } else if (ev.type === 'coords') {
          const cached = molByUuid.get(ev.slot)
          if (!cached) return // we don't know the topology yet; topology event will arrive
          const allCoords = await fetchCoords(ev.coords_url)
          const mol: MoleculeKitDict = { ...cached, coords: allCoords, numFrames: ev.numFrames }
          molByUuid.set(ev.slot, mol)
          await v.updateSlotCoords(ev.slot, mol)
        } else if (ev.type === 'remove') {
          molByUuid.delete(ev.slot)
          await v.removeSlot(ev.slot)
          setSlots((prev) => prev.filter((s) => s.uuid !== ev.slot))
        }
      },
      setStatus,
    )
    return close
  }, [])

  const onToggle = (uuid: string) => {
    setSlots((prev) =>
      prev.map((s) => {
        if (s.uuid !== uuid) return s
        const next = { ...s, visible: !s.visible }
        viewerRef.current?.setSlotVisibility(uuid, next.visible)
        return next
      }),
    )
  }

  const onDelete = async (uuid: string) => {
    try {
      await fetch(`/unregister/${uuid}`, { method: 'POST' })
    } catch {
      // Slot will linger; the user can refresh.
    }
  }

  return (
    <div style={{ position: 'fixed', inset: 0 }}>
      <MolstarViewer ref={viewerRef} />
      <SlotSidebar slots={slots} onToggle={onToggle} onDelete={onDelete} />
      <StatusBanner status={status} />
    </div>
  )
}

function SlotSidebar({
  slots,
  onToggle,
  onDelete,
}: {
  slots: SlotState[]
  onToggle: (uuid: string) => void
  onDelete: (uuid: string) => void
}) {
  // Default position is bottom-right so we don't sit on top of mol*'s
  // top-left viewport controls (animation + trajectory + snapshots).
  const [pos, setPos] = useState<{ right: number; bottom: number }>({ right: 12, bottom: 12 })
  const dragRef = useRef<{ startX: number; startY: number; startRight: number; startBottom: number } | null>(null)

  if (slots.length === 0) return null

  const onHeaderMouseDown = (e: ReactMouseEvent) => {
    e.preventDefault()
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startRight: pos.right,
      startBottom: pos.bottom,
    }
    const onMove = (ev: MouseEvent) => {
      const d = dragRef.current
      if (!d) return
      // Right/bottom anchored: dragging right *decreases* `right`, dragging down *decreases* `bottom`.
      const right = Math.max(0, d.startRight - (ev.clientX - d.startX))
      const bottom = Math.max(0, d.startBottom - (ev.clientY - d.startY))
      setPos({ right, bottom })
    }
    const onUp = () => {
      dragRef.current = null
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  return (
    <div
      style={{
        position: 'absolute',
        right: pos.right,
        bottom: pos.bottom,
        background: 'rgba(20,20,28,0.85)',
        color: '#eee',
        font: '13px/1.4 system-ui, sans-serif',
        borderRadius: 6,
        minWidth: 180,
        zIndex: 10,
        userSelect: 'none',
      }}
    >
      <div
        onMouseDown={onHeaderMouseDown}
        style={{
          fontWeight: 600,
          padding: '6px 10px',
          cursor: 'move',
          borderBottom: '1px solid rgba(255,255,255,0.1)',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
        title="Drag to move"
      >
        <span style={{ opacity: 0.6, fontSize: 11 }}>⋮⋮</span>
        Molecules
      </div>
      <div style={{ padding: '6px 10px' }}>
        {slots.map((s) => (
          <div
            key={s.uuid}
            style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '2px 0' }}
          >
            <input
              type="checkbox"
              checked={s.visible}
              onChange={() => onToggle(s.uuid)}
            />
            <span style={{ flex: 1, opacity: s.visible ? 1 : 0.5 }}>{s.label}</span>
            <button
              onClick={() => onDelete(s.uuid)}
              style={{
                background: 'transparent',
                color: '#eee',
                border: '1px solid #555',
                borderRadius: 3,
                padding: '0 6px',
                cursor: 'pointer',
              }}
              title="Remove from viewer"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

function StatusBanner({ status }: { status: SSEStatus }) {
  if (status.kind === 'open') return null
  const text =
    status.kind === 'closed-stale'
      ? 'Server restarted — refresh page'
      : `Connection lost (retry ${status.retries})`
  return (
    <div
      style={{
        position: 'absolute',
        bottom: 12,
        right: 12,
        background: 'rgba(180,40,40,0.9)',
        color: 'white',
        font: '13px/1.4 system-ui, sans-serif',
        padding: '6px 10px',
        borderRadius: 4,
        zIndex: 20,
      }}
    >
      {text}
    </div>
  )
}
