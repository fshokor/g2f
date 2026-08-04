#!/usr/bin/env bash
# Run from WSL2, inside the repo: bash scripts/download_g2f_2024.sh
# Downloads into data/raw/Training_data/, verifying each file isn't the
# CyVerse HTML landing page before moving on to the next.
set -uo pipefail  # no -e: one file's failure must not kill the whole run

BASE_URL="https://de.cyverse.org/anon-files//iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Training_data"
OUT_DIR="data/raw/Training_data"
mkdir -p "$OUT_DIR"

FILES=(
  "1_Training_Trait_Data_2014_2023.csv"
  "2_Training_Meta_Data_2014_2023.csv"
  "3_Training_Soil_Data_2015_2023.csv"
  "4_Training_Weather_Data_2014_2023_full_year.csv"
  "4_Training_Weather_Data_2014_2023_seasons_only.csv"
  "5_Genotype_Data_All_2014_2025_Hybrids.vcf"
  "5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt"
  "6_Training_EC_Data_2014_2023.csv"
  "key_inbreds_G2F_2014-2025.txt"
)

FAILED=()

for f in "${FILES[@]}"; do
  echo "Downloading $f ..."
  # -C - resumes a partial download instead of restarting from zero.
  # --retry 5 with backoff handles transient 503s from the server.
  if curl -fSL -C - --retry 5 --retry-delay 3 --retry-all-errors \
       -o "$OUT_DIR/$f" "$BASE_URL/$f"; then
    if head -c 200 "$OUT_DIR/$f" | grep -qi '<!DOCTYPE html'; then
      echo "  WARNING: $f looks like an HTML page, not real data."
      FAILED+=("$f")
    else
      size=$(stat -c '%s' "$OUT_DIR/$f")
      echo "  OK -- ${size} bytes"
    fi
  else
    echo "  FAILED after retries: $f"
    FAILED+=("$f")
  fi
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All files downloaded successfully."
else
  echo "${#FAILED[@]} file(s) still failed -- re-run this script to retry just those,"
  echo "curl will resume rather than restart thanks to -C -:"
  printf '  %s\n' "${FAILED[@]}"
fi

echo
echo "Also grab readme.txt manually from the same folder if it exists --"
echo "it wasn't in this file list and needs checking before any loader code is written."