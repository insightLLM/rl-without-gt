#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /checkpoint_load/<model_name> [comma_separated_step_ids]"
  exit 1
fi

SRC_ROOT="$1"
STEP_IDS="${2:-}"     # e.g. "70,80,90"
DEST_ROOT="$3"
MERGER_SCRIPT="$4"

echo "🔍 Starting migration from: $SRC_ROOT"
echo "📂 Destination root: $DEST_ROOT"
echo "🔧 Merger script: $MERGER_SCRIPT"
[[ -n "$STEP_IDS" ]] && echo "🔢 Only processing steps: $STEP_IDS"

IFS=',' read -ra IDS_ARRAY <<< "$STEP_IDS"

for STEP_DIR in "$SRC_ROOT"/global_step_*; do
  [[ -d "$STEP_DIR" ]] || continue
  STEP_NAME=$(basename "$STEP_DIR")
  STEP_NUM="${STEP_NAME#global_step_}"

  # skip if user specified list and this step isn't in it
  if [[ -n "$STEP_IDS" && ! " ${IDS_ARRAY[*]} " =~ " ${STEP_NUM} " ]]; then
    echo "⏭ Skipping $STEP_NAME (not in list)"
    continue
  fi

  TARGET="$DEST_ROOT/$STEP_NAME"
  echo -e "\n➡️ Processing $STEP_NAME"
  mkdir -p "$TARGET"
  echo "  Copying actor files..."
  cp -rv "$STEP_DIR/actor/"* "$TARGET/"

  echo "  Running model merger..."
  if python "$MERGER_SCRIPT" --local_dir "$TARGET"; then
    echo "    ✔ Merge succeeded"
  else
    echo "    ❌ Merge failed"
  fi

  pushd "$TARGET" > /dev/null
    echo "  Removing intermediate .pt files..."
    rm -fv extra_* model_* optim_*

    if [[ -d huggingface ]]; then
      echo "  Moving huggingface/* → $TARGET"
      mv -v huggingface/* .
      rmdir huggingface && echo "    Removed empty huggingface directory"
    else
      echo "  No huggingface directory to move"
    fi
  popd > /dev/null

  echo "✅ Finished $STEP_NAME"
done

echo -e "\n🎉 All specified global_step directories processed."
