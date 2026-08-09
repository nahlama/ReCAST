#!/usr/bin/env bash
set -euo pipefail

workspace="${1:-/mnt/d/My-Work/Esophageal/New}"
target="$workspace/Data/GENCODE/v50/gencode.v50.transcripts.fa.gz"
expected_sha256="5a320f524d73b5793518eb19b118829033713443d0f42af20a67bb31cc06cf56"
url="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.transcripts.fa.gz"

mkdir -p "$(dirname "$target")"
if [[ ! -f "$target" ]]; then
  curl -L --fail --retry 3 --output "$target" "$url"
fi
printf '%s  %s\n' "$expected_sha256" "$target" | sha256sum --check --status
printf 'Verified GENCODE 50 transcript FASTA: %s\n' "$target"
