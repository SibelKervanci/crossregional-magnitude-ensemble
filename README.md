# crossregional-magnitude-ensemble

Code accompanying the manuscript

> Kervancı, İ.S., Tezel, İ. *Cross-benchmark leakage and cross-regional generalization in machine-learning earthquake magnitude estimation: a harmonized multi-dataset workflow with external validation in Türkiye.* Submitted to *Computers & Geosciences*.

The workflow harmonizes three open seismic waveform benchmarks (STEAD, INSTANCE, CEED) in physical units, audits them for shared events, extracts 32 physically motivated features, trains individual learners and ensembles (averaging, blending, stacking, KNORA-E/U), handles network-specific S-phase availability with pattern submodels, and evaluates the models within each benchmark, across benchmarks and, without adaptation, on manually picked records of the Kandilli Observatory (KOERI) from south-eastern Türkiye.

## Requirements

- Python 3.11 (the full environment is given in `environment.yml`)
- A multi-core CPU and at least 64 GB RAM (128 GB recommended); no GPU is required
- About 1 TB of disk space for the three benchmarks

```bash
conda env create -f environment.yml
conda activate deprem
```

All paths and settings are defined in **`config.py`**. Edit `PROJECT`, `ROOT` and `KOERI` before running anything. Every script reads its defaults from this file, so the scripts can be run without arguments (for example with F5 in Spyder) or from the command line; command-line arguments override the defaults.

## Data

| Data | Source |
|---|---|
| STEAD, INSTANCE, CEED | Downloaded through [SeisBench](https://github.com/seisbench/seisbench) by `download_datasets.py`. If the SeisBench download fails on your system, `manual_download.py` downloads the same files directly and resumes interrupted downloads. |
| Station metadata (instrument sensitivities) | FDSN web services, retrieved by `fetch_station_gains.py` |
| KOERI earthquake catalogue | Earthquake query service of the KOERI Regional Earthquake-Tsunami Monitoring Center (http://www.koeri.boun.edu.tr); save the result as a text file in the `KOERI` folder and set `KOERI_CATALOG` in `config.py` |
| KOERI reviewed phase bulletins (GSE2.0) | Monthly/yearly phase-reading files published by KOERI (http://www.koeri.boun.edu.tr/sismo/2/bultenler/aylik-bulten-faz-okumalari/faz_okumalari/). Access requires a password that KOERI provides on request; the files are **not** redistributed here. Save the yearly files as `KOERI/faz/<year>.txt`. |
| KO network waveforms | KOERI FDSN data centre (network KO, https://doi.org/10.7914/SN/KO), retrieved by `build_turkey_testset.py` |

## Workflow

Run the scripts in the following order. Intermediate products are written to disk, and every step resumes where it stopped if interrupted.

### 1. Training benchmarks

| Script | Purpose |
|---|---|
| `download_datasets.py` | Download STEAD and INSTANCE (`--stage 1`) and CEED (`--stage 2`) |
| `manual_download.py` | Alternative, resumable download of the SeisBench files |
| `check_ceed_components.py` | Verify the component order of CEED waveforms |
| `fetch_station_gains.py` | Retrieve instrument sensitivities for STEAD and INSTANCE |
| `build_trace_index.py` | Select records ($M_L$, ≤ 110 km, velocity sensors, ≤ 5 stations per event), assign sensitivities, create event-level splits |
| `audit_leakage.py` | Find events shared between benchmarks and events in the external test region; writes the exclusion list |
| `extract_features.py` | Convert to velocity, filter 1–20 Hz and compute the 32 features (use `--workers 1` on Windows if parallel processing stalls) |

### 2. External test set (Türkiye)

| Script | Purpose |
|---|---|
| `parse_koeri_phases.py` | Read the GSE2.0 phase bulletins and match them with the catalogue |
| `check_turkey_stations.py` | List KO stations with velocity sensors |
| `build_turkey_testset.py` | Download KO waveforms around the manual P picks and compute the same features |

### 3. Experiments

| Script | Purpose |
|---|---|
| `train_models.py` | Train and evaluate all learners and ensembles for one configuration |
| `run_experiments.py` | Run all experiments: `--set core` (main experiment, leave-one-dataset-out and cross-domain tuning), `--set nosp` (S–P-free pattern submodels), `--set ablations` |
| `tune_ensembles.py` | Cross-domain (leave-one-domain-out) selection of ensemble and KNORA parameters, with routing |

Recommended order: `run_experiments.py --set core`, then `--set nosp`, then `tune_ensembles.py`, then `--set ablations`.

### 4. Evaluation, figures and tables

| Script | Purpose |
|---|---|
| `evaluate_turkey.py` | Zero-shot evaluation in Türkiye, including routed predictions, magnitude bins, SNR bands and few-shot offset calibration |
| `evaluate_routed_test.py` | Routed evaluation of the in-region and cross-dataset test sets |
| `analyze_missing_s.py` | Errors and bias for records with and without an S arrival |
| `make_results.py` | Main tables, paired tests, Friedman–Nemenyi analysis and all figures of the manuscript |
| `shap_analysis.py` | SHAP feature importance and its transferability across regions |
| `make_supplement.py` | Supplementary tables S2–S5 |

Shared settings and helper functions are in `config.py` and `common.py`.

## Reproducibility notes

- All splits are made at the **event level**; all traces of an event belong to the same subset.
- The leakage audit removes duplicate events across benchmarks (79,826 STEAD–CEED and 565 STEAD–INSTANCE events in our run) and all events in the external test region before training.
- Hyperparameters are fixed and identical in all experiments; the random seed is set in `config.py`.
- Test data, including the Türkiye test set, are never used for training, model selection or tuning.
- Runtime on a 20-core workstation with 128 GB RAM: feature extraction about 1.7 h; main experiment about 2.5 h.

## Citation

If you use this code, please cite the article above (citation details will be updated after publication) and the original datasets: STEAD (Mousavi et al., 2019), INSTANCE (Michelini et al., 2021), CEED (Zhu et al., 2025), SeisBench (Woollam et al., 2022) and the KO network (https://doi.org/10.7914/SN/KO).

## License

MIT License (see `LICENSE`).

## Contact

İlkay Sibel Kervancı, Department of Computer Engineering, Gaziantep University — skervanci@gantep.edu.tr
