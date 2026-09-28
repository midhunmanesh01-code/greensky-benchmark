# GreenSky Benchmark (SIH26081)

A quantitative benchmarking and machine-learning blending framework for daily precipitation forecasting in India. This repository evaluates and enhances numerical weather prediction (NWP) models (ECMWF IFS025 and NOAA GFS Seamless) against ground-truth India Meteorological Department (IMD) rain gauge observations.

---

## 📁 Repository Structure

```
greensky-benchmark/
├── data/                                 # Raw input data and cached atmospheric features
│   ├── ecmwf_vs_imd.csv                  # Model vs IMD comparison dataset
│   ├── experiment4_atmos_cache.csv       # Cached Open-Meteo atmospheric variables
│   ├── imdweb_hMXvt0uE.csv               # Raw IMD observation records
│   ├── test_data.csv                     # Evaluation test split
│   └── train_data.csv                    # Training split
│
├── results/                              # Benchmark metrics, predictions, and diagnostics
│   ├── adaptive_benchmark_results.csv    # GBR adaptive blender test predictions
│   ├── adaptive_weight_results.csv       # Daily optimal weight optimization results
│   ├── benchmark_results.csv             # Baseline daily model predictions & blends
│   ├── corrected_benchmark_results.csv   # Convention B (03:00 UTC aligned) benchmark results
│   ├── corrected_diagnostic.csv          # Window alignment diagnostics per IMD date
│   ├── experiment2_results.csv           # Temporal robustness test predictions
│   ├── experiment3_results.csv           # Extreme precipitation calibration results
│   ├── experiment4_results.csv           # Multi-variable atmospheric feature models
│   ├── gating_benchmark_results.csv      # Decision tree gating blender predictions
│   └── regime_blender_results.csv        # Regime-based adaptive blender predictions
│
├── scripts/                              # Benchmark, blending, and experiment scripts
│   ├── adaptive_blender.py               # GBR blender with dynamic weight learning
│   ├── alignment_check.py                # IMD timestamp and data continuity verification
│   ├── benchmark.py                      # Original baseline benchmark (Convention A)
│   ├── corrected_benchmark.py            # Corrected benchmark (Convention B / 03:00 UTC)
│   ├── experiment2_robustness.py         # Exp 2: Robustness evaluation across time windows
│   ├── experiment3_extreme_correction.py # Exp 3: Extreme precipitation calibration
│   ├── experiment4_atmospheric_context.py# Exp 4: Multi-variable atmospheric context ML
│   ├── gating_blender.py                 # Decision-tree gating blender
│   └── regime_blender.py                 # Exp 1: Rainfall intensity regime blender
│
└── README.md
```

---

## ⚙️ Core Concepts & Alignment

IMD daily rainfall observations represent 24-hour accumulated rainfall measured at **08:30 IST (03:00 UTC)** on date $D$, covering $(D-1)\text{ 03:00 UTC} \to D\text{ 03:00 UTC}$.

- **Convention A (Original Benchmark)**: Aggregates model forecasts on calendar UTC days ($00:00 \to 23:00\text{ UTC}$).
- **Convention B (Corrected Benchmark)**: Corrects the accumulation window to match IMD ground truth ($03:00\text{ UTC} \to 03:00\text{ UTC}$), eliminating the 5.5-hour phase discrepancy.

---

## 🔬 Scripts & Pipeline Overview

| Script | Description | Primary Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| [`alignment_check.py`](scripts/alignment_check.py) | Verifies IMD gauge data intervals, duplicates, and timestamp continuity | `data/imdweb_hMXvt0uE.csv` | Console stdout |
| [`benchmark.py`](scripts/benchmark.py) | Baseline benchmark evaluating ECMWF, GFS, Equal Blend, and Gradient Boosting Regressor under UTC daily aggregation | `data/imdweb_hMXvt0uE.csv`, Open-Meteo API | `results/adaptive_benchmark_results.csv` |
| [`adaptive_blender.py`](scripts/adaptive_blender.py) | Optimizes daily blend weights minimizing individual day error, predicting optimal weights via GBR | `results/benchmark_results.csv` | `results/adaptive_weight_results.csv` |
| [`gating_blender.py`](scripts/gating_blender.py) | Decision-tree based classifier for model routing based on forecast divergence and seasonal indicators | `results/benchmark_results.csv` | `results/gating_benchmark_results.csv` |
| [`corrected_benchmark.py`](scripts/corrected_benchmark.py) | Full benchmark under IMD-aligned accumulation windows (Convention B) | `data/imdweb_hMXvt0uE.csv`, `results/benchmark_results.csv`, Open-Meteo API | `results/corrected_benchmark_results.csv`, `results/corrected_diagnostic.csv` |
| [`regime_blender.py`](scripts/regime_blender.py) | **Experiment 1**: Splits forecasts into precipitation regimes (Dry, Light, Moderate, Heavy) and learns optimal weights per regime | `results/corrected_benchmark_results.csv` | `results/regime_blender_results.csv` |
| [`experiment2_robustness.py`](scripts/experiment2_robustness.py) | **Experiment 2**: Tests model stability with inverted/alternative chronological splits | `results/corrected_benchmark_results.csv` | `results/experiment2_results.csv` |
| [`experiment3_extreme_correction.py`](scripts/experiment3_extreme_correction.py) | **Experiment 3**: Post-processing calibration and scaling for extreme and heavy rainfall events | `results/corrected_benchmark_results.csv` | `results/experiment3_results.csv` |
| [`experiment4_atmospheric_context.py`](scripts/experiment4_atmospheric_context.py) | **Experiment 4**: Integrates atmospheric features (CAPE, RH, geopotential height, wind speed) from NWP reanalysis | `results/corrected_benchmark_results.csv`, `data/experiment4_atmos_cache.csv` | `results/experiment4_results.csv` |

---

## 🚀 How to Run

Each script is configured using relative directory resolution (`pathlib.Path`), allowing execution from both the repository root or within the `scripts/` folder:

```bash
# Run baseline alignment check
python scripts/alignment_check.py

# Run corrected benchmark
python scripts/corrected_benchmark.py

# Run Experiment 1 (Regime Blender)
python scripts/regime_blender.py

# Run Experiment 2 (Robustness)
python scripts/experiment2_robustness.py

# Run Experiment 3 (Extreme Correction)
python scripts/experiment3_extreme_correction.py

# Run Experiment 4 (Atmospheric Context)
python scripts/experiment4_atmospheric_context.py
```

---

## 📊 Evaluation Metrics

The benchmark measures forecast skill using:
- **MAE (Mean Absolute Error)**: Average magnitude of absolute errors (mm).
- **RMSE (Root Mean Squared Error)**: Penalizes larger forecast discrepancies (mm).
- **CSI (Critical Success Index / Threat Score)**: Evaluates categorical detection accuracy across thresholds ($15.6\text{ mm/day}$ and $64.5\text{ mm/day}$):
  $$\text{CSI} = \frac{\text{Hits}}{\text{Hits} + \text{Misses} + \text{False Alarms}}$$
