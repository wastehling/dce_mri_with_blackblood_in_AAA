# Black-blood DCE-MRI of Abdominal Aortic Aneurysms

Code used to evaluate the black-blood dynamic contrast-enhanced (DCE) MRI
protocol described in:

> Stehling W, Aalbregt E, Wennen M, Schrauben E, Yeung KK, Nederveen AJ,
> Gurney-Champion OJ. Black-blood dynamic contrast-enhanced MRI of
> abdominal aortic aneurysms. Magn Reson Mater Phy (2026).
> https://doi.org/10.1007/s10334-026-01386-z

Because the underlying patient data cannot be shared, this repository ships
with a script that generates a **synthetic (pseudo) dataset** with the same
folder structure and file naming as the real data, so the evaluation and
statistics pipeline can be run end to end by anyone who clones this repo.

**The simulated data is not physiologically realistic** — it is a hollow
cylinder ("vessel wall") with a made-up wash-in/washout curve, plus a
blood-suppressed "lumen" core, only meant to exercise the code. Any numbers
it produces (diameters, growth rates, DCE parameters, SNR/CNR, statistics)
are meaningless and only demonstrate that the pipeline runs.
`data_predefined/diameter_prestudy.csv` is likewise entirely fabricated,
not derived from any real patient measurements.

## Installation

```bash
pip install -r requirements.txt
```

## Running the pipeline

Run these in order from the repository root:

```bash
python simulated_dce_data.py   # writes reconstructed_images/ and masks/
python compute_annual_growth.py   # writes data_processed/*.pkl from data_predefined/diameter_prestudy.csv
python main_eval_study.py      # evaluates each simulated scan, writes evaluation/
python main_statistics.py      # test-retest and association statistics/plots
```

All paths are configured in `evaluation.yml` (relative to the repo root by
default). Generated data, evaluation results, and pickled intermediates are
not tracked in git — re-run the steps above to (re)create them.

## Repository layout

- `simulated_dce_data.py` — generates the pseudo dataset (NIfTI DCE volumes + masks).
- `data_predefined/diameter_prestudy.csv` — pseudo pre-study diameter measurements.
- `compute_annual_growth.py` — computes pre-study AAA growth rates.
- `main_eval_study.py` — per-patient DCE evaluation (Kalifa parameters, SNR/CNR).
- `main_statistics.py` — test-retest and association statistics.
- `calculations.py`, `data_loading.py`, `helpers_*.py`, `plotting.py`, `utilities.py` — supporting modules.
