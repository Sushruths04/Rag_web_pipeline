import { useRef, useState } from 'react'

interface Props {
  files: File[]
  onFiles: (files: File[]) => void
}

export default function UploadDropzone({ files, onFiles }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)

  const accept = (list: FileList | null) => {
    if (!list) return
    const pdfs = [...list].filter((f) => f.name.toLowerCase().endsWith('.pdf'))
    onFiles([...files, ...pdfs])
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          accept(e.dataTransfer.files)
        }}
        style={{
          border: `1.5px dashed ${over ? 'var(--accent)' : 'var(--line)'}`,
          borderRadius: 12,
          padding: '38px 20px',
          textAlign: 'center',
          cursor: 'pointer',
          background: over ? 'color-mix(in srgb, var(--accent) 6%, transparent)' : 'transparent',
          transition: 'border-color 0.2s, background 0.2s',
        }}
      >
        <div style={{ fontSize: 15, fontWeight: 500 }}>Drop PDFs here</div>
        <div className="dim" style={{ marginTop: 4, fontSize: 12 }}>
          or click to browse — any structure: standards, textbooks, reports
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        multiple
        hidden
        onChange={(e) => {
          accept(e.target.files)
          e.target.value = ''
        }}
      />
      {files.length > 0 && (
        <ul style={{ listStyle: 'none', marginTop: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {files.map((f, i) => (
            <li key={`${f.name}-${i}`} className="mono" style={{ fontSize: 12, display: 'flex', justifyContent: 'space-between', padding: '4px 8px', background: 'var(--raised)', borderRadius: 6 }}>
              <span>{f.name}</span>
              <span className="dim">
                {(f.size / 1024 / 1024).toFixed(2)} MB{' '}
                <button
                  style={{ border: 'none', background: 'none', color: 'var(--err)', padding: '0 0 0 8px', fontSize: 12 }}
                  onClick={() => onFiles(files.filter((_, j) => j !== i))}
                  aria-label={`Remove ${f.name}`}
                >
                  ✕
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
