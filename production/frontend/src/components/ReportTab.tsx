import { artifactUrl } from '../lib/api'
import type { ArtifactRow } from '../lib/types'

interface Props {
  runId: string
  artifacts: ArtifactRow[]
}

export default function ReportTab({ runId, artifacts }: Props) {
  const report = artifacts.find((a) => a.name === 'report.html')

  if (!report)
    return (
      <div className="empty-state">
        The evaluation report is generated after the final eval run — retrieval quality against the
        ground truth, per-question drill-down, and the cost/latency comparison. Nothing to show yet.
      </div>
    )

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div style={{ padding: '8px 16px', display: 'flex', justifyContent: 'flex-end' }}>
        <a href={`${artifactUrl(runId, report.name)}?download=1`} download>
          <button>Download report</button>
        </a>
      </div>
      <iframe
        title="Evaluation report"
        src={artifactUrl(runId, report.name)}
        sandbox="allow-same-origin"
        style={{ flex: 1, border: 'none', background: '#fff' }}
      />
    </div>
  )
}
