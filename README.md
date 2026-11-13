# VLMF

Validation-Locked Missingness-Conditioned Fusion (VLMF) for intrusion detection
with incomplete network-flow telemetry.

VLMF keeps two fitted predictors fixed, an XGBoost model and a schema-aware
neural predictor (SchemaFlow), and selects on a validation partition how their
probabilities are combined for each declared field-loss condition. The selected
policies are committed before the test partition is opened.

## What this release contains

- `flowfm/` — cohort construction, field-loss mask generation, the mask
  resolver, validation-locked policy selection, and the predictors.
- `scripts/` — the runners for the reported experiments, including
  `run_cross_dataset_formal_matrix.py` (original cohorts) and
  `run_supplementary_matrix.py` (resampled cohorts).
- `configs/` — the experiment configuration files.
- `manifests/datasets/` — provenance and checksums for the source datasets.
- `tests/` — the test suite (`python -m unittest discover -s tests -v`).

The source datasets are not included. See below for how to obtain them.

## Reproducing the reported experiments

### Datasets

The experiments use three public NetFlow-v2 datasets released by the
University of Queensland. Each `manifests/datasets/*.source.json` records the
dataset DOI, citation, and checksums.

| Dataset | Split seed (original cohort) |
|---|---|
| NF-BoT-IoT-v2 | 20260727 |
| NF-CSE-CIC-IDS2018-v2 | 20260731 |
| NF-ToN-IoT-v2 | 20260729 |

NF-UNSW-NB15-v2 (split seed 20260728; restart cohort seed 20260801) is used
only as a pretraining source and is not a reported evaluation dataset.

Place each dataset under the data root at the path recorded in its manifest.
The manifests record the dataset DOI, citation, and the checksums of the
files as acquired.

### Splits

Each cohort is a group-disjoint 0.6/0.2/0.2 split of one collection, drawn with
the split seed above. The three resampled cohorts per dataset repeat that split
with seeds 20261018, 20261019, and 20261020. These are resampled splits of the
same three collections, not independent environments.

### Seeds

- Endpoint models in the original cohorts were fit with seeds 17, 42, and
  20260725 (`scripts/run_cross_dataset_formal_matrix.py`).
- The resampled cohorts used seed 17 (`scripts/run_supplementary_matrix.py`).
- The setting-level bootstrap used seed 20260730 with 20,000 resamples.
- The cluster bootstrap used 2,000 resamples.

### Software

Python 3.10.8, PyTorch 2.1.2+cu118, NumPy 1.26.4, pandas 2.2.2,
scikit-learn 1.5.1, XGBoost 2.0.3, and LightGBM 4.6.0. Pinned versions are in
`constraints-autodl.txt` and the `requirements-*.txt` files.

### Running

The runners read data from `/root/autodl-tmp` by default, which is the data disk
of the server used for the reported runs. Point them at your own data directory
by editing the `_DATA_ROOT` assignment at the top of the runner, or by passing
the data root argument where the runner accepts one. Nothing else in the
pipeline depends on that path: the frozen protocol hashes cover the method and
the evaluation conditions, not file locations.

Neural training was run without deterministic cuDNN settings, so rerunning it
reproduces the procedure, configuration, and seeds but not bitwise-identical
predictions. XGBoost and LightGBM training uses the histogram method with a fixed seed.

## Scope

The field-loss masks are synthetic. They follow declared field sets, including
documented exporter profiles, and do not replay exporter or parser failures
recorded in operation. The repository reports the conditions under which VLMF
trails a baseline as well as those under which it leads.

## License

Released under the MIT License. See `LICENSE`.
