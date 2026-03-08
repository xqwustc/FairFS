# FairFS

Implementation of "FairFS: Addressing Deep Feature Selection Biases for Recommender System," published at `The Web Conference 2026`.

## Introduction

FairFS is an innovative feature selection algorithm designed to address three significant biases:

- **Layer Bias** by regularizing feature importance across all non-linear transformational layers.
- **Baseline Bias** and **Approximation Bias** by using a smooth baseline feature close to the classifier's decision boundary alongside an aggregated approximation method.

These methodologies ensure that FairFS provides a more accurate feature selection process.

## Datasets

The original datasets used are provided in the links below (as mentioned in the associated paper):

- **Criteo**: [Access the Criteo Display Ad Challenge Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/)
- **Avazu**: [Access the Avazu CTR Prediction Dataset](https://kaggle.com/competitions/avazu-ctr-prediction)
- **iFly-AD**: [Access the iFly-AD CTR Forecast Challenge Dataset](https://challenge.xfyun.cn/topic/info?type=CTR-forecastoption=ssgy)

### Processed Datasets

Pre-processed versions of these datasets can be found at:

- [FuxiCTR on GitHub](https://github.com/xue-pai/FuxiCTR)
- [BARS on GitHub](https://github.com/reczoo/BARS)

## Workflow

First clone FuxiCTR, then replace the `model_zoo` folder in the FuxiCTR repository with the contents of this repo.

The FairFS pipeline consists of three sequential steps: **Pretrain → Eval → Incre**.

### Step 1: Pretrain

**Command**: `python run_expid_embigNorm_select.py --stage pretrain`

#### Purpose

Train the model on all features to obtain checkpoints under different `--normk` (regularization coefficient λ) values for the subsequent Eval stage.

#### Key Arguments

| Argument | Description |
|----------|-------------|
| `--expid` | Experiment ID; must match a config entry |
| `--normk` | Regularization coefficient k; supports multiple values, e.g. `--normk 0 0.01 0.1 1 10 100 1000` |
| `--interp` | Interpolation count (INTERP); used in Eval; can be fixed during pretrain |
| `--baseline` | Baseline type; default `smooth` |
| `--gpu` | GPU device ID |

#### Output

- Model checkpoints saved under `checkpoints/{expid}/`
- Naming format: `expid={expid}_k={normk}_baseline={baseline}.model`

---

### Step 2: Feature Field Importance Evaluation

**Command**: `python run_expid_embigNorm_select.py --stage eval`

#### Purpose

Load checkpoints from the Pretrain stage and compute feature importance on the validation set, producing `feature_importance_result_k={k}_baseline={baseline}_itp={interp}.csv`.

#### Key Arguments

| Argument | Description |
|----------|-------------|
| `--expid` | Same as in Pretrain |
| `--normk` | Same as or a subset of Pretrain values |
| `--interp` | Interpolation count (INTERP); multiple values can be run separately |
| `--baseline` | Baseline type |
| `--gpu` | GPU device ID |

#### Output

- Feature importance CSV: `feature_importance_result_k={k}_baseline=smooth_itp={interp}.csv`
- Stored in `checkpoints/{expid}/`

---

### Step 3: Top-K Feature Field Re-training

**Command**: `python run_expid_incre.py`

#### Purpose

Using the feature importance files from Step 2, incrementally train and evaluate the model for increasing Top-K feature counts, recording LogLoss and AUC for each configuration. Results are written to `incre_eval_results.log`.

#### Key Arguments

| Argument | Description |
|----------|-------------|
| `--expid` | Experiment ID |
| `--gpu` | GPU device ID |
| `--feat_file` | Path to feature importance file (optional; if omitted, all `feature_importance_result_k*.csv` in the checkpoint dir are processed) |
| `--K` | Number of top features (num_features) for re-training. When set, overrides the default i_list logic and evaluates only this single K. |

#### Output

- Results appended to `checkpoints/{expid}/incre_eval_results.log`
- Line format: `{file_basename}, num_features={n}, logloss={x}, AUC={y}, train_time={t}s, inf_time={i}s`

---

## Suggested Hyperparameters

Recommended commands for 4 models × 3 datasets. Parameter mapping: `--normk` corresponds to NORMK_VALUES (k), `--interp` corresponds to INTERP (itp). The `--K` in the re-train step (Step 3) should match the `num_features` for each configuration.

### DCN

**DCN-Criteo**

```bash
cd model_zoo/DCN/DCN_torch
python run_expid_embigNorm_select.py --expid DCN_criteo_pre --normk 10.0 --interp 5 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid DCN_criteo_pre --normk 10.0 --interp 5 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid DCN_criteo_pre --gpu 0 --feat_file checkpoints/DCN_criteo_pre/feature_importance_result_k=10.0_baseline=smooth_itp=5.csv --K 36
```

**DCN-Avazu**

```bash
cd model_zoo/DCN/DCN_torch
python run_expid_embigNorm_select.py --expid DCN_avazu_pre --normk 0.01 --interp 10 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid DCN_avazu_pre --normk 0.01 --interp 10 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid DCN_avazu_pre --gpu 0 --feat_file checkpoints/DCN_avazu_pre/feature_importance_result_k=0.01_baseline=smooth_itp=10.csv --K 15
```

**DCN-iFly**

```bash
cd model_zoo/DCN/DCN_torch
python run_expid_embigNorm_select.py --expid DCN_iflychu_pre --normk 100.0 --interp 10 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid DCN_iflychu_pre --normk 100.0 --interp 10 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid DCN_iflychu_pre --gpu 0 --feat_file checkpoints/DCN_iflychu_pre/feature_importance_result_k=100.0_baseline=smooth_itp=10.csv --K 200
```

### WideDeep

**WideDeep-Criteo**

```bash
cd model_zoo/WideDeep/WideDeep_torch
python run_expid_embigNorm_select.py --expid WideDeep_criteo_pre --normk 0.01 --interp 5 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid WideDeep_criteo_pre --normk 0.01 --interp 5 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid WideDeep_criteo_pre --gpu 0 --feat_file checkpoints/WideDeep_criteo_pre/feature_importance_result_k=0.01_baseline=smooth_itp=5.csv --K 36
```

**WideDeep-Avazu**

```bash
cd model_zoo/WideDeep/WideDeep_torch
python run_expid_embigNorm_select.py --expid WideDeep_avazu_pre --normk 0.01 --interp 5 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid WideDeep_avazu_pre --normk 0.01 --interp 5 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid WideDeep_avazu_pre --gpu 0 --feat_file checkpoints/WideDeep_avazu_pre/feature_importance_result_k=0.01_baseline=smooth_itp=5.csv --K 16
```

**WideDeep-iFly**

```bash
cd model_zoo/WideDeep/WideDeep_torch
python run_expid_embigNorm_select.py --expid WideDeep_iflychu_pre --normk 100.0 --interp 5 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid WideDeep_iflychu_pre --normk 100.0 --interp 5 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid WideDeep_iflychu_pre --gpu 0 --feat_file checkpoints/WideDeep_iflychu_pre/feature_importance_result_k=100.0_baseline=smooth_itp=5.csv --K 100
```

### DeepFM

**DeepFM-Criteo**

```bash
cd model_zoo/DeepFM/DeepFM_torch
python run_expid_embigNorm_select.py --expid DeepFM_criteo --normk 0.01 --interp 5 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid DeepFM_criteo --normk 0.01 --interp 5 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid DeepFM_criteo --gpu 0 --feat_file checkpoints/DeepFM_criteo/feature_importance_result_k=0.01_baseline=smooth_itp=5.csv --K 36
```

**DeepFM-Avazu**

```bash
cd model_zoo/DeepFM/DeepFM_torch
python run_expid_embigNorm_select.py --expid DeepFM_avazu --normk 100.0 --interp 1 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid DeepFM_avazu --normk 100.0 --interp 1 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid DeepFM_avazu --gpu 0 --feat_file checkpoints/DeepFM_avazu/feature_importance_result_k=100.0_baseline=smooth_itp=1.csv --K 20
```

**DeepFM-iFly**

```bash
cd model_zoo/DeepFM/DeepFM_torch
python run_expid_embigNorm_select.py --expid DeepFM_iflychu --normk 10.0 --interp 10 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid DeepFM_iflychu --normk 10.0 --interp 10 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid DeepFM_iflychu --gpu 0 --feat_file checkpoints/DeepFM_iflychu/feature_importance_result_k=10.0_baseline=smooth_itp=10.csv --K 150
```

### FM

**FM-Criteo**

```bash
cd model_zoo/FM
python run_expid_embigNorm_select.py --expid FM_criteo --normk 10.0 --interp 10 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid FM_criteo --normk 10.0 --interp 10 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid FM_criteo --gpu 0 --feat_file checkpoints/FM_criteo/feature_importance_result_k=10.0_baseline=smooth_itp=10.csv --K 36
```

**FM-Avazu**

```bash
cd model_zoo/FM
python run_expid_embigNorm_select.py --expid FM_avazu --normk 100.0 --interp 10 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid FM_avazu --normk 100.0 --interp 10 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid FM_avazu --gpu 0 --feat_file checkpoints/FM_avazu/feature_importance_result_k=100.0_baseline=smooth_itp=10.csv --K 17
```

**FM-iFly**

```bash
cd model_zoo/FM
python run_expid_embigNorm_select.py --expid FM_iflychu --normk 0.1 --interp 5 --baseline smooth --gpu 0 --stage pretrain
python run_expid_embigNorm_select.py --expid FM_iflychu --normk 0.1 --interp 5 --baseline smooth --gpu 0 --stage eval
python run_expid_incre.py --expid FM_iflychu --gpu 0 --feat_file checkpoints/FM_iflychu/feature_importance_result_k=0.1_baseline=smooth_itp=5.csv --K 170
```

---

## Notes

1. **Execution order**: Run Pretrain → Eval → Incre in sequence. Eval depends on Pretrain checkpoints; Incre depends on Eval feature importance files.
2. **Hyperparameter search**: `--normk` and `--interp` accept multiple values; the program iterates over combinations. For long runs, validate with a single configuration first.
3. **Incre resumption**: `run_expid_incre.py` checks `incre_eval_results.log` and skips existing `(file, num_features)` entries, enabling resumable runs.
4. **Model layout**: DCN, WideDeep, and DeepFM use `model_zoo/{Model}/{Model}_torch/`; FM uses `model_zoo/FM/`. Execute commands from the corresponding directory.

---

## Acknowledgements

Our development is based on [FuxiCTR](https://github.com/reczoo/FuxiCTR). We appreciate their contributions.
