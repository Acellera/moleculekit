export type SSEStatus =
  | { kind: 'open' }
  | { kind: 'closed-stale' }       // server returned 410 — refresh page
  | { kind: 'lost'; retries: number }

export function connect(
  session: string,
  onEvent: (data: unknown) => void,
  onStatus: (status: SSEStatus) => void,
): () => void {
  let retries = 0
  let es: EventSource | null = null
  let closed = false

  const open = () => {
    es = new EventSource(`/events?session=${encodeURIComponent(session)}`)
    es.onopen = () => {
      retries = 0
      onStatus({ kind: 'open' })
    }
    es.onmessage = (ev) => {
      try {
        onEvent(JSON.parse(ev.data))
      } catch (e) {
        console.error('SSE parse error', e)
      }
    }
    es.onerror = () => {
      // EventSource doesn't surface status codes. Probe with fetch to
      // distinguish 410 (stale session) from a transient network drop.
      fetch(`/events?session=${encodeURIComponent(session)}`, {
        headers: { Accept: 'text/event-stream' },
      }).then((r) => {
        if (r.status === 410) {
          es?.close()
          closed = true
          onStatus({ kind: 'closed-stale' })
        } else {
          retries += 1
          onStatus({ kind: 'lost', retries })
          if (!closed && retries < 100 && es?.readyState === EventSource.CLOSED) {
            setTimeout(open, Math.min(5000, 500 * retries))
          }
        }
      }).catch(() => {
        retries += 1
        onStatus({ kind: 'lost', retries })
      })
    }
  }

  open()
  return () => {
    closed = true
    es?.close()
  }
}
