# FairTrade - HiWi Challenge Submission

This branch (`hiwi-challenge-submission`) contains modifications to the original
[L3S/FairTrade](https://github.com/L3S/FairTrade) codebase made as part of the
"Ensuring Fairness in Federated Learning" HiWi selection challenge (L3S Research
Center, Leibniz Universität Hannover).

All modifications are isolated to this branch; `main` remains identical to the
original upstream repository.

## What's in this branch

Two files were modified relative to the original repo:

- **`FairTrade.py`**: environment/library compatibility fixes, reproducibility
  (fixed seed), model checkpoint saving, and the Task 3 multi-attribute fairness
  extension (gender + nationality).
- **`load_data_utilities.py`**: one-line fix for a `psmpy` API rename
  (`knn_matched` → `kdtree_matched`).

See **Bug Fixes Applied** and **Task 3 Extension** below for exactly what changed
and why.

## Setup

Tested on Google Colab (Python 3.12, CPU runtime). From a fresh environment:

```bash
git clone https://github.com/zeinabsadat/FairTrade.git
cd FairTrade
git checkout hiwi-challenge-submission
pip install -r requirements.txt
```

> Note: the original `requirements.txt` pins exact versions (e.g. `torch==2.0.1`)
> that are no longer available on current package indexes. The versions listed
> now are unpinned and were confirmed to install and run correctly as of this
> submission (see **Bug Fixes Applied**, Fix 1).

## Bug Fixes Applied

Running the original codebase out-of-the-box failed for several
environment/library-compatibility reasons unrelated to the paper's methodology.
These were fixed as follows (all in `FairTrade.py` unless noted):

| # | Issue | Fix |
|---|-------|-----|
| 1 | `device = torch.device('mps')` only works on Apple Silicon | Changed to `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| 2 | `from botorch.fit import fit_gpytorch_model` (function removed in current botorch) | Import removed (unused; `fit_gpytorch_mll` is used elsewhere in the file) |
| 3 | `evaluate()` return-value mismatch: called expecting 3 return values, function returns 1 | Fixed both call sites to `objectives = evaluate(...)` |
| 4 | `psmpy`'s `PsmPy.knn_matched` renamed to `kdtree_matched` in current `psmpy` versions (`load_data_utilities.py`) | Updated call site accordingly |
| 5 | No fixed random seed, results not reproducible across runs | Added `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` at the top of the script |
| 6 | Trained model was never saved to disk, so it could not be reloaded for Task 2/3 analysis | Added `torch.save(global_model.state_dict(), "results/adult/global_model_final.pt")` right after the final `global_model.eval()` |

## How to Run Task 1 (Reproduce & Understand)

```bash
mkdir -p results/adult results/bank results/default results/law results/kdd

python FairTrade.py \
  --fairness_notion 'stat_parity' \
  --num_clients 3 \
  --dataset_name 'adult' \
  --epochs 15 \
  --communication_rounds 50 \
  --mobo_optimization_rounds 10 \
  --distribution_type 'random'
```

This trains FairTrade on the Adult dataset with gender as the sensitive
attribute, reproducing the setup used for Table 1 of the original paper. Results
(`balanced accuracy`, `statistical parity`) are printed each communication round
and saved to `results/adult/3_bal_acc_stat_parity.npy` and
`results/adult/3_stat_parity.npy`. The trained model is saved to
`results/adult/global_model_final.pt`.

**Important:** back up the result files before running Task 3, since Task 3 uses
the same output filenames and will overwrite them:

```bash
cp results/adult/global_model_final.pt results/adult/global_model_task1_task2.pt
cp results/adult/3_bal_acc_stat_parity.npy results/adult/3_bal_acc_stat_parity_task1.npy
cp results/adult/3_stat_parity.npy results/adult/3_stat_parity_task1.npy
```

## How to Run Task 2 (Intersectional Fairness Evaluation)

Task 2 is a post-hoc analysis of the Task 1 model; it does not require
retraining. After Task 1 has run at least once (so
`results/adult/global_model_final.pt` exists), run the following in a Python
session from the repo root:

```python
import torch, pandas as pd
from torch import nn
from load_data_utilities import load_dataset

def create_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1), nn.Sigmoid()
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clients_data, X_test, y_test, sex_list, column_names_list, ytest_potential = load_dataset(
    'datasets/adult.csv', 'adult', 3, 'sex', 'random'
)
X_test, y_test = X_test.to(device), y_test.to(device)

global_model = create_model(X_test.shape[1]).to(device)
global_model.load_state_dict(torch.load('results/adult/global_model_final.pt', map_location=device))
global_model.eval()

with torch.no_grad():
    y_pred_cls = global_model(X_test).squeeze().round()

race_idx = column_names_list.index('race')
sex_idx = column_names_list.index('sex')
race_list = X_test[:, race_idx].cpu().numpy()
sex_list_arr = X_test[:, sex_idx].cpu().numpy()
y_pred_np = y_pred_cls.cpu().numpy()

def compute_spd(y_pred, group_values, privileged_value):
    priv_mask = (group_values == privileged_value)
    unpriv_mask = (group_values != privileged_value)
    return y_pred[priv_mask].mean() - y_pred[unpriv_mask].mean(), priv_mask.sum(), unpriv_mask.sum()

spd_gender, n_m, n_f = compute_spd(y_pred_np, sex_list_arr, privileged_value=1.0)
spd_race, n_w, n_nw = compute_spd(y_pred_np, race_list, privileged_value=4.0)  # White -> 4 per LabelEncoder

is_white = (race_list == 4.0)
is_male = (sex_list_arr == 1.0)
groups = {
    "White Male":       y_pred_np[is_white & is_male],
    "White Female":     y_pred_np[is_white & ~is_male],
    "Non-White Male":   y_pred_np[~is_white & is_male],
    "Non-White Female": y_pred_np[~is_white & ~is_male],
}
rates = {name: preds.mean() for name, preds in groups.items()}
max_group, min_group = max(rates, key=rates.get), min(rates, key=rates.get)
spd_intersectional = rates[max_group] - rates[min_group]

print(f"SPD (gender): {spd_gender:.4f}")
print(f"SPD (race, White vs Non-White): {spd_race:.4f}")
print(f"Intersectional SPD (max gap): {spd_intersectional:.4f} ({max_group} vs {min_group})")
```

SPD (Statistical Parity Difference) is computed manually per Verma & Rubin
(2018), the same definition used as Equation 2 in the FairTrade paper. Race is
binarized as White (LabelEncoder value `4`) vs. Non-White, consistent with the
four intersectional subgroups the task specifies (White Male, White Female,
Non-White Male, Non-White Female).

## How to Run Task 3 (Second Sensitive Attribute: Nationality)

Task 3 extends FairTrade to optimize fairness jointly for gender and
nationality (`native.country`, binarized as United-States vs. Non-US). This
required modifying `FairTrade.py` directly (already applied on this branch);
the key additions are:

- A second `DemographicParityLoss` instance for nationality, summed with the
  original gender loss: `fairness_loss = fairness_loss_sex + NAT_WEIGHT * fairness_loss_nat`
  (`NAT_WEIGHT = 1.0`).
- The MOBO fairness objective now uses
  `combined_spd = max(abs(stat_parity_sex), abs(stat_parity_nationality))`,
  the worst-case SPD across both attributes, rather than gender SPD alone.

Run it with the same command as Task 1; the script now trains against both
attributes automatically:

```bash
python FairTrade.py \
  --fairness_notion 'stat_parity' \
  --num_clients 3 \
  --dataset_name 'adult' \
  --epochs 15 \
  --communication_rounds 50 \
  --mobo_optimization_rounds 10 \
  --distribution_type 'random'
```

Training output now prints both `statistical parity (sex)` and
`statistical parity (nationality)` each round. As with Task 1, back up the
resulting model and result arrays before running anything else:

```bash
cp results/adult/global_model_final.pt results/adult/global_model_task3.pt
cp results/adult/3_bal_acc_stat_parity.npy results/adult/3_bal_acc_stat_parity_task3.npy
cp results/adult/3_stat_parity.npy results/adult/3_stat_parity_task3.npy
```

The design rationale, trade-offs, and comparison against Task 1's single-attribute
model are discussed in the PDF report (Task 3 section).

## Notes on Reproducibility

- A fixed seed (42) is set for `torch`, `numpy`, and `random`, but minor
  run-to-run variation may still occur due to non-deterministic operations
  inside BoTorch/GPyTorch (e.g. Monte Carlo sampling in the acquisition
  function). This does not affect the qualitative conclusions reported.
- All CPU-only; no GPU is required, though the code will use CUDA automatically
  if available.

## Repository Structure

```
FairTrade/
├── FairTrade.py              # main training script (modified, see Bug Fixes / Task 3 above)
├── load_data_utilities.py    # data loading + causal fairness matching (modified, Fix 4)
├── constraint.py             # fairness loss definitions (unmodified)
├── utilities.py               # evaluation metrics (unmodified)
├── datasets/                 # Adult, Bank, Default, Law, KDD datasets (unmodified)
├── requirements.txt          # updated, unpinned dependency list
└── results/                  # created at runtime; stores trained models and metrics
```
