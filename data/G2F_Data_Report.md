# Data Report: Genomes to Fields (G2F) Genotype-by-Environment Prediction Competition 2025

## Overview

This folder contains the training and testing data for the G2F maize genotype-by-environment (G×E) yield prediction competition. The goal is to predict `Yield_Mg_ha` for hybrids grown in held-out 2024 environments, using historical (2014–2023) trait, environmental, soil, weather, and genotype data. Data is provided "as is" — the source documentation (`COMPETITION_DATA_README.docx`) explicitly notes it is only lightly filtered and contains substantial missing data by design (e.g., local checks and commercial hybrids have trait values deliberately set to NA).

Two top-level folders:
- **Training_data/** — 2014–2023 (genotype file extends to 2025 hybrids)
- **Testing_data/** — 2024 only

Each file is linked to the others via an `Env` column (location + year) and/or `Hybrid` column (hybrid name).

---

## Training Data

### 1. Trait Data — `1_Training_Trait_Data_2014_2023.csv`
Plot-level phenotypic and yield records — the core training target table.

- **Shape:** 173,960 rows × 26 columns
- **Types:** 3 object (categorical/text), 2 int64, 21 float64
- **Keys:** 272 unique `Env`, 5,205 unique `Hybrid`, years 2014–2023
- **Missing data (top columns):** Root_Lodging_plants 38.1%, Stalk_Lodging_plants 30.4%, Silk_DAP_days 24.2%, Pollen_DAP_days 22.8%, Twt_kg_m3 21.4%, Plant_Height_cm 12.8%, Ear_Height_cm 11.9%, Range 10.8%, Pass 10.6%, Stand_Count_plants 9.9%, Grain_Moisture 5.4%, Yield_Mg_ha 5.2% (the prediction target itself is ~5% missing in training). No missing values in Env, Year, Hybrid, Plot, Replicate, Block, Experiment, Field_Location.
- **Describe (numeric highlights):** Year spans 2014–2023 (mean ~2019); Yield_Mg_ha ranges up to 23.3 Mg/ha (mean ~9.6); Plant_Height_cm up to 350 cm; Twt_kg_m3 mean ~732.
- **No fully duplicated rows.**
- **Note:** Per the README, missingness here is partly intentional QC filtering (e.g., implausible height/yield/moisture values nulled out) plus "Local check"/commercial hybrids having trait values set to NA.

### 2. Metadata — `2_Training_Meta_Data_2014_2023.csv`
Per-environment (location/year) site metadata.

- **Shape:** 272 rows × 38 columns
- **Types:** 27 object, 10 float64, 1 int64
- **Keys:** 272 unique Env (1:1 with rows), 10 unique years
- **Missing data:** Heavily missing free-text/optional fields — Issue/comment #6–#8 (96–99.6%), Irrigated (81.6%), Issue/comment #2–#5 (76–97%), Soil_Taxonomic_ID description (71.7%), In-season tillage method (68.4%), Trial_ID (46.3%), Cardinal_Heading_Pass_1 (39.7%). Core identifying fields (Env, Year) are complete.
- **No duplicate rows.**

### 3. Soil Data — `3_Training_Soil_Data_2015_2023.csv`
Per-environment soil chemistry/texture (no 2014 data collected).

- **Shape:** 186 rows × 36 columns
- **Types:** 29 float64, 6 object, 1 int64
- **Keys:** 186 unique Env, 9 unique years
- **Missing data:** Trace micronutrients (Copper, Iron, Boron, Zinc, Manganese ppm) missing ~98.4%; Comments 97.3%; BpH 94.1%; texture-related fields (% Silt/Sand/Clay, Texture) ~11–13%; several chemistry fields (lbs N/A, Salts, WDRF Buffer pH, CEC) ~10.2%.
- **No duplicate rows.**

### 4. Weather Data — `4_Training_Weather_Data_2014_2023_full_year.csv` / `..._seasons_only.csv`
Daily weather (NASA POWER) per Env.

- **Full year shape:** 98,236 rows × 18 columns; 269 unique Env
- **Seasons-only shape:** 51,098 rows × 18 columns (subset: 14 days before planting to 14 days after harvest); 269 unique Env
- **Types:** 16 float64, 1 object (Env), 1 int64 (Date, as YYYYMMDD integer)
- **Missing data:** None in either file (0% across all columns) — this is the cleanest table in the dataset.
- **Columns:** Env, Date, plus 16 daily weather variables (temperature min/max/mean, dew point, wet bulb temp, humidity, precipitation, solar radiation, surface pressure, wind speed, soil moisture/wetness proxies).

### 5. Genotype Data — `5_Genotype_Data_All_2014_2025_Hybrids.vcf` and `..._numerical.txt`
SNP-level genotype data covering both training and testing hybrids combined (2014–2025).

- **VCF file:** standard VCF v4.0, 2,425 variant records (rows) × 5,899 hybrid samples (columns beyond the 9 standard VCF fields). ~55 MB.
- **Numerical file:** tab-delimited, 5,899 hybrids (rows) × 2,425 SNP markers (+1 label column = 2,426 columns), ~39 MB. Genotype values encoded as 0 / 0.5 / 1 (dosage-style) with `NA` for missing calls.
- **Supporting file — `key_inbreds_G2F_2014-2025.txt`:** 2,660 inbred lines with columns Cultivar, Dataset, SourceName, Bioproject, BioSample, Alternative name, Comments — maps genotype names to sequencing technology/source (GBS, skim-seq, exome capture, DNBSEQ across different year ranges) and public accession IDs.
- **Note from README:** Not all hybrids in the trait data have genotype data (commercial hybrids or QC-failed samples excluded); genotype matrix is not full rank (fewer SNPs than hybrids); minimally filtered, expect errors — QC recommended before use.

### 6. Environmental Covariates — `6_Training_EC_Data_2014_2023.csv`
Derived environmental covariates (crop-model-based) per Env.

- **Shape:** 241 rows × 655 columns
- **Types:** 636 float64, 18 int64, 1 object (Env)
- **Missing data:** None (0% across all columns).
- **Structure:** 673 covariates named as `<variable>_<phenological period>_<soil layer 1-10>` (e.g., `SDR_pGerEme_1`), covering variables like water supply-demand ratio, thermal time, biomass, evapotranspiration, LAI, soil water, heat-stress day counts, etc., across 9 phenological periods and up to 10 soil depth layers (20cm each, 2m total).

---

## Testing Data (2024)

### 1. Submission Template — `1_Submission_Template_2024.csv`
Defines exactly which Env/Hybrid combinations require yield predictions.

- **Shape:** 10,057 rows × 3 columns (Env, Hybrid, Yield_Mg_ha)
- **Types:** 2 object, 1 float64
- **Missing data:** Yield_Mg_ha is 100% missing (by design — this is what participants must predict). Env and Hybrid fully populated.
- **Keys:** 23 unique Env, 1,063 unique Hybrid.

### 2. Metadata — `2_Testing_Meta_Data_2024.csv`
Same schema family as training metadata, for 2024 only.

- **Shape:** 23 rows × 40 columns (2 extra columns vs. training: Date_Planted and Plot_Area_ha are included here at the Env level, since 2024 plot-level trait data isn't released)
- **Types:** 23 object, 16 float64, 1 int64
- **Missing data:** Similar pattern to training — Issue/comment #3–#8 missing 91–100%, In-season tillage method 73.9%, Soil_Taxonomic_ID description 65.2%, Field 39.1%, Pre-plant tillage/weather station fields ~35%.
- **Keys:** 23 unique Env (1:1 with rows).

### 3. Soil Data — `3_Testing_Soil_Data_2024.csv`
- **Shape:** 21 rows × 35 columns
- **Types:** 16 float64, 14 int64, 5 object
- **Missing data:** Comments 100%, trace micronutrients (Zn, Mn, Cu, Fe) and BpH ~95.2%, texture fields (%Silt/Sand/Clay, Texture) 4.8%.
- **Keys:** 16 unique Env (fewer than the 21 rows — some Env have multiple soil samples, or duplicate Env entries).

### 4. Weather Data — `4_Testing_Weather_Data_2024_full_year.csv` / `..._seasons_only.csv`
- **Full year shape:** 7,245 rows × 18 columns; 23 unique Env
- **Seasons-only shape:** 4,591 rows × 18 columns; 23 unique Env
- **Missing data:** None — complete, same 18-column schema as training weather files.

### 5. Environmental Covariates — `6_Testing_EC_Data_2024.csv`
- **Shape:** 22 rows × 655 columns (same 673-covariate schema as training)
- **Types:** matches training EC file structure
- **Missing data:** None.
- **Keys:** 22 unique Env.

### 6. Observed Values (holdout ground truth) — `7_Testing_Observed_Values.csv`
- **Shape:** 9,486 rows × 3 columns (Env, Hybrid, Yield_Mg_ha)
- **Types:** 2 object, 1 float64
- **Missing data:** None — this file is fully populated (likely released post-competition as ground truth for the submission template rows).
- **Keys:** 22 unique Env, 1,063 unique Hybrid.

Note: the genotype VCF/numerical files in `Training_data/` already cover both training and testing hybrids (2014–2025), so there is no separate testing genotype file.

---

## Cross-Cutting Observations

- **Linking key:** `Env` (location × year) and `Hybrid` (hybrid name, itself a cross of two inbred parents) are the join keys across almost every file. The README warns these don't always match perfectly across files — reconciliation is a modeling decision left to participants.
- **Missingness is structural, not random** in several places: comment/issue fields and rare micronutrients are missing >90% because they were rarely recorded, not because of data corruption. Trait QC also intentionally nulls out biologically implausible values (documented thresholds in the README, e.g., flowering data nulled when environment means fall below 45 days).
- **Cleanest tables:** weather (both training and testing) and environmental covariates (EC) — 0% missing in all cases.
- **Sparsest tables:** metadata "Issue/comment" fields and soil micronutrients — both dominated by missing values (>90%).
- **Scale mismatch:** the trait file (173,960 plot-level rows) is far larger than the number of unique Env×Hybrid combinations, since each Hybrid can appear across multiple plots/replicates within an Env.
- **Genotype data caveats (per README):** minimally filtered, contains known errors, not full rank, and doesn't cover all phenotyped hybrids (commercial checks / QC failures excluded).
- **Target variable:** `Yield_Mg_ha` — present with 5.2% missing in the training trait data, 100% missing (to be predicted) in the submission template, and fully present in the testing observed-values ground-truth file.

---

## File Size Summary

| File | Size |
|---|---|
| Training_data/1_Training_Trait_Data_2014_2023.csv | 31 MB |
| Training_data/2_Training_Meta_Data_2014_2023.csv | 100 KB |
| Training_data/3_Training_Soil_Data_2015_2023.csv | 28 KB |
| Training_data/4_Training_Weather_Data_2014_2023_full_year.csv | 10 MB |
| Training_data/4_Training_Weather_Data_2014_2023_seasons_only.csv | 5.3 MB |
| Training_data/5_Genotype_Data_All_2014_2025_Hybrids.vcf | 55 MB |
| Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt | 39 MB |
| Training_data/6_Training_EC_Data_2014_2023.csv | 1.9 MB |
| Training_data/key_inbreds_G2F_2014-2025.txt | 228 KB |
| Testing_data/1_Submission_Template_2024.csv | 332 KB |
| Testing_data/2_Testing_Meta_Data_2024.csv | 12 KB |
| Testing_data/3_Testing_Soil_Data_2024.csv | 4 KB |
| Testing_data/4_Testing_Weather_Data_2024_full_year.csv | 688 KB |
| Testing_data/4_Testing_Weather_Data_2024_seasons_only.csv | 416 KB |
| Testing_data/6_Testing_EC_Data_2024.csv | 188 KB |
| Testing_data/7_Testing_Observed_Values.csv | 412 KB |
