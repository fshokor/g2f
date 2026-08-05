#!/usr/bin/env bash
# Run from WSL2, inside the repo: bash scripts/download_g2f_2024.sh
# Downloads Training_data/ and Testing_data/ into data/raw/, verifying each
# file isn't the CyVerse HTML landing page before moving on to the next.
# File names confirmed directly against the CyVerse folder listings on
# 2025-08-05 -- update DOI_ROOT below if CyVerse ever restructures the release.
set -uo pipefail  # no -e: one file's failure must not kill the whole run

DOI_ROOT="https://de.cyverse.org/anon-files//iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025"

TRAINING_FILES=(
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

# No genotype file here -- the single VCF/numerical file above in Training_data
# already covers all 2014-2025 hybrids and is shared across both splits.
TESTING_FILES=(
  "1_Submission_Template_2024.csv"
  "2_Testing_Meta_Data_2024.csv"
  "3_Testing_Soil_Data_2024.csv"
  "4_Testing_Weather_Data_2024_full_year.csv"
  "4_Testing_Weather_Data_2024_seasons_only.csv"
  "6_Testing_EC_Data_2024.csv"
  "7_Testing_Observed_Values.csv"
)

FAILED=()

download_folder () {
  local folder_name="$1"
  shift
  local files=("$@")
  local base_url="$DOI_ROOT/$folder_name"
  local out_dir="data/raw/$folder_name"
  mkdir -p "$out_dir"

  echo "=== $folder_name ==="
  for f in "${files[@]}"; do
    echo "Downloading $f ..."
    # -C - resumes a partial download instead of restarting from zero.
    # --retry 5 with backoff handles transient 503s from the server
    # (curl honors any Retry-After header the server sends, e.g. 300s).
    if curl -fSL -C - --retry 5 --retry-delay 3 --retry-all-errors \
         -o "$out_dir/$f" "$base_url/$f"; then
      if head -c 200 "$out_dir/$f" | grep -qi '<!DOCTYPE html'; then
        echo "  WARNING: $f looks like an HTML page, not real data."
        FAILED+=("$folder_name/$f")
      else
        size=$(stat -c '%s' "$out_dir/$f")
        echo "  OK -- ${size} bytes"
      fi
    else
      echo "  FAILED after retries: $f"
      FAILED+=("$folder_name/$f")
    fi
  done
  echo
}

download_folder "Training_data" "${TRAINING_FILES[@]}"
download_folder "Testing_data" "${TESTING_FILES[@]}"

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All files downloaded successfully."
else
  echo "${#FAILED[@]} file(s) still failed -- re-run this script to retry just those,"
  echo "curl will resume rather than restart thanks to -C -:"
  printf '  %s\n' "${FAILED[@]}"
fi

echo
echo "Also grab readme.txt manually from the same folder if it exists --"
echo "it wasn't in either listing and needs checking before any loader code is written."