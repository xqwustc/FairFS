# FairFS - KDD'25 ADS Repo

## Introduction

FairFS is an innovative feature selection algorithm designed to address three significant biases:

- **Layer Bias** by regularizing feature importance across all non-linear transformational layers.
- **Baseline Bias** and **Approximation Bias** by using a smooth baseline feature close to the classifier’s decision boundary alongside an aggregated approximation method.

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

## Usage Example

Below is an example of how to use FairFS with the DeepFM model:

```bash
python /DeepFM/DeepFM_torch/run_expid_FairFS.py --expid DeepFM_[dataset] --gpu 0 --normk [loss_strength] --nanchor [n_anchor] --baseline [mean/zero/sample_mean]
```

Replace [dataset], [loss_strength], [n_anchor], and [mean/zero/sample_mean] with actual values to match your specific experimental setup.

## Acknowledgements

Our development is based on the work of [FuxiCTR](https://github.com/reczoo/FuxiCTR). We appreciate their contributions.

