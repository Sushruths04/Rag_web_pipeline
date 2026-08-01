#!/usr/bin/env bash
# Max-yield rerun of all 5 DIN/ISO standards after the 2026-08-01 GT quality fixes.
#
# Fresh --out-dir per doc: no Stage-3 chunk cache is reused, so the extraction
# fixes (".." joiner, four-line watermark strip) actually take effect. Reusing
# the old cache would silently replay pre-fix fact text.
#
# PYTHONPATH is pinned to THIS repo's src/ -- without it the monorepo engine
# wins the editable install and the run silently uses different code.
set -u

REPO="D:/Mini Thesis/Rag_web_pipeline"
export PYTHONPATH="${REPO}/src"
export PYTHONIOENCODING=utf-8
OUT="${REPO}/data/reruns_postfix"
mkdir -p "$OUT"

run () {
  local id="$1" pdf="$2"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] START $id"
  echo "=============================================================="
  python -X faulthandler -m rag_gt.allpdf.pipeline \
    --doc-id "$id" \
    --pdf "$pdf" \
    --out-dir "$OUT/$id" \
    --max-pairs 800 \
    --target-chains 40 \
    --multihop-chains 40 \
    --multihop-depths 2 \
    --score-necessity \
    --docling-page-cap 60 \
    > "$OUT/$id.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $id (exit $?)"
  tail -5 "$OUT/$id.log"
}

run din_iso_3834_1  "${REPO}/rag_files/DIN EN ISO 3834-1-ENG.pdf"
run din_iso_4136    "${REPO}/rag_files/DIN EN ISO 4136-ENG.pdf"
run din_iso_13919_1 "${REPO}/rag_files/DIN EN ISO 13919-1-ENG.pdf"
run din_iso_3452_1  "${REPO}/rag_files/DIN EN ISO 3452-1-ENG.pdf"
run din_iso_6507_1  "${REPO}/rag_files/DIN EN ISO 6507-1-ENG.pdf"

echo "ALL DONE"
