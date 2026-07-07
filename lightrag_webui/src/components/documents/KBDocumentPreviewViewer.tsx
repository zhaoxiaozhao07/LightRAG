import { useEffect, useMemo, useState } from 'react'
import {
  downloadKBDocumentPreviewFallback,
  getKBDocumentPreviewVariantBlob,
  getKBDocumentPreviewVariantText,
  KBDocumentPreviewFallback,
  KBDocumentPreviewManifest,
  KBDocumentPreviewVariant
} from '@/api/lightrag'

type TablePreviewPayload = {
  kind?: string
  source_name?: string
  truncated?: boolean
  sheets?: Array<{
    name: string
    rows: string[][]
  }>
}

type KBDocumentPreviewViewerProps = {
  manifest: KBDocumentPreviewManifest
  className?: string
}

const downloadBlob = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

const fallbackFilename = (
  manifest: KBDocumentPreviewManifest,
  fallback: KBDocumentPreviewFallback
) => manifest.source_name || `${fallback.artifact_id}`

const isBlobVariant = (variant: KBDocumentPreviewVariant) =>
  variant.media_type.startsWith('image/') || variant.media_type === 'application/pdf'

export default function KBDocumentPreviewViewer({
  manifest,
  className
}: KBDocumentPreviewViewerProps) {
  const variant = manifest.preferred ?? manifest.variants[0] ?? null
  const [text, setText] = useState<string | null>(null)
  const [blobUrl, setBlobUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const tablePayload = useMemo<TablePreviewPayload | null>(() => {
    if (!text || variant?.kind !== 'table') {
      return null
    }
    try {
      return JSON.parse(text) as TablePreviewPayload
    } catch {
      return null
    }
  }, [text, variant?.kind])

  useEffect(() => {
    let cancelled = false
    let currentBlobUrl: string | null = null

    if (!variant) {
      return () => undefined
    }

    const load = async () => {
      setLoading(true)
      setText(null)
      setBlobUrl(null)
      setError(null)
      try {
        if (isBlobVariant(variant)) {
          const blob = await getKBDocumentPreviewVariantBlob(variant)
          if (cancelled) {
            return
          }
          currentBlobUrl = URL.createObjectURL(blob)
          setBlobUrl(currentBlobUrl)
        } else {
          const responseText = await getKBDocumentPreviewVariantText(variant)
          if (!cancelled) {
            setText(responseText)
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Preview failed')
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    load()

    return () => {
      cancelled = true
      if (currentBlobUrl) {
        URL.revokeObjectURL(currentBlobUrl)
      }
    }
  }, [variant])

  const handleDownloadFallback = async () => {
    if (!manifest.fallback) {
      return
    }
    const blob = await downloadKBDocumentPreviewFallback(manifest.fallback)
    downloadBlob(blob, fallbackFilename(manifest, manifest.fallback))
  }

  const fallbackButton = manifest.fallback ? (
    <button
      type="button"
      className="rounded border px-3 py-1 text-sm hover:bg-muted"
      onClick={handleDownloadFallback}
    >
      Download original
    </button>
  ) : null

  if (!variant) {
    return (
      <div className={className}>
        <div className="rounded border border-dashed p-4 text-sm text-muted-foreground">
          No safe inline preview is available for this document.
          <div className="mt-3">{fallbackButton}</div>
        </div>
      </div>
    )
  }

  return (
    <div className={className}>
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium">{manifest.source_name}</div>
          <div className="text-xs text-muted-foreground">
            {variant.kind} · {variant.media_type}
          </div>
        </div>
        {fallbackButton}
      </div>

      {loading ? (
        <div className="rounded border p-4 text-sm text-muted-foreground">Loading preview...</div>
      ) : null}
      {error ? (
        <div className="rounded border border-destructive/40 p-4 text-sm text-destructive">
          {error}
        </div>
      ) : null}

      {!loading && !error && variant.kind === 'table' && tablePayload ? (
        <div className="overflow-auto rounded border">
          {(tablePayload.sheets ?? []).map((sheet) => (
            <div key={sheet.name} className="min-w-full">
              <div className="border-b bg-muted px-3 py-2 text-sm font-medium">
                {sheet.name}
              </div>
              <table className="w-full border-collapse text-sm">
                <tbody>
                  {sheet.rows.map((row) => {
                    const rowKey = row.map((cell, index) => `${index}:${cell}`).join('|')
                    const cellCounts = new Map<string, number>()
                    return (
                      <tr key={rowKey}>
                        {row.map((cell) => {
                          const occurrence = cellCounts.get(cell) ?? 0
                          cellCounts.set(cell, occurrence + 1)
                          return (
                            <td key={`${rowKey}:${cell}:${occurrence}`} className="border px-2 py-1 align-top">
                              {cell}
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      ) : null}

      {!loading && !error && variant.kind === 'html' && text ? (
        <iframe
          title={`Preview ${manifest.source_name}`}
          className="h-[70vh] w-full rounded border"
          sandbox=""
          srcDoc={text}
        />
      ) : null}

      {!loading && !error && variant.kind !== 'table' && variant.kind !== 'html' && text ? (
        <pre className="max-h-[70vh] overflow-auto whitespace-pre-wrap rounded border bg-muted/30 p-4 text-sm">
          {text}
        </pre>
      ) : null}

      {!loading && !error && blobUrl && variant.media_type.startsWith('image/') ? (
        <img
          src={blobUrl}
          alt={`Preview ${manifest.source_name}`}
          className="max-h-[70vh] max-w-full rounded border object-contain"
        />
      ) : null}

      {!loading && !error && blobUrl && variant.media_type === 'application/pdf' ? (
        <object
          data={blobUrl}
          type="application/pdf"
          className="h-[70vh] w-full rounded border"
        >
          PDF preview is unavailable in this browser.
        </object>
      ) : null}
    </div>
  )
}
