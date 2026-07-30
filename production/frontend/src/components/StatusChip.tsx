export default function StatusChip({ status }: { status: string }) {
  return <span className={`chip ${status}`}>{status}</span>
}
