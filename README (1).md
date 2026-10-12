# Observational Fidelity for synthetic linkage data

Code for the paper *Observational Fidelity: calibrating and checking synthetic data as a
proxy for record linkage quality without labelled data* (Lam, Mantovani, Pearson,
Christen and Harron; submitted to BMC Medical Research Methodology).

The `ofeval` package builds synthetic replicas of a name-based linkage task and checks
whether they can stand in for the real task when true links are unknown. It:

- **synthesises clean names** by swapping surnames within strata, keeping observed name
  frequencies and, optionally, the sizes of shared full-name clusters;
- **learns an error profile** from labelled pairs (supervised), from high-scoring
  candidate pairs (pseudo-truth), or by **calibration**: indirect inference that tunes the
  error profile until replicas, viewed through the same biased learner, resemble the
  observed task, optionally with a small clerical sample;
- **assesses each generator** with a label-free **fidelity gate** that compares
  diagnostics of the observed task and its replicas against replica-to-replica noise and
  stated margins;
- **evaluates linkage quality** (precision, missed-match rate, group differences) at a
  threshold fixed on the observed task.

## Installation

Requires Python 3.11 or later.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # Windows: .venv\Scripts\pip install -e ".[dev]"
pytest
```

The tests use small synthetic data and do not need the input files below.

## Data

Input data are not included. See [`data/README.md`](data/README.md) for the files
and where to obtain them, and place them in `data/raw/`.

## Reproducing the paper

Run the commands from the repository root. Times are approximate, on a laptop with
four workers.

| Paper | Command | Output |
|---|---|---|
| Tables 4 and 5, Figures 2, 3 and 5 (full-scale analysis; ~30 min) | `ofeval run --study configs/study_scale.yaml --workers 4` | `outputs/study_scale/` |
| Table 6, Figure 4 (redraw study; ~3 h) | `python scripts/calibration_sweep.py --study configs/study_scale.yaml --out outputs/calibration_sweep_scale --blocks 5 --sizes 0,200 --seeds 10 --workers 4` | `outputs/calibration_sweep_scale/` |
| Fidelity gate operating characteristics (10,000-record tasks) | `python scripts/gate_operating_characteristics.py --blocks 10 --workers 4` | `outputs/gate_operating_characteristics/` |
| Additional file 1, Table S1 (synthesisers at full scale) | `python scripts/scale_synthesis_check.py --workers 4` | `outputs/scale_synthesis_check/` |
| Full-name duplication and cluster sizes | `python scripts/name_duplication_check.py` | `outputs/name_duplication_check/` |
| Near-identical names and false-match decomposition (run the full-scale analysis first) | `python scripts/near_duplicate_check.py` | `outputs/near_duplicate_check/` |
| 10,000-record analysis | `ofeval run --study configs/study.yaml --workers 4` | `outputs/study/` |
| Preliminary analysis and name-sampler comparison | `ofeval run --study configs/paper.yaml --workers 4` and `python scripts/synthesis_comparison.py` | `outputs/paper/`, `outputs/synthesis_comparison/` |
| Figures 2–5 (after the runs above) | `python scripts/manuscript_figures.py` | `outputs/manuscript_figures/` |
| Figure 1 (literature map; data in `docs/literature_map.csv`) | `python scripts/literature_map.py` | `outputs/manuscript_figures/` |

`ofeval run` writes long-format results (`long/`), summary tables (`tables/`) and
figures (`figures/`) for each study. `ofeval summarise --study <config>` rebuilds the
tables and figures from existing long-format results.

## Layout

```
configs/           study configurations (full scale, 10,000 records, preliminary) and scenarios
src/ofeval/        package: names, synthesis, linkage, profiles, calibration, diagnostics, gate, experiment
scripts/           analyses and figures reported in the paper
docs/              literature map data (Figure 1, Additional file 1 Table S3)
tests/             unit and integration tests
```

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper once published.

## Licence

[To add before publishing the repository.]
