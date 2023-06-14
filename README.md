# 📄 *Improving reduced-order models through nonlinear decoding of projection-dependent model outputs*

## Data

Datasets used in this study are stored in `data/` directory. These include combustion datasets for:

- hydrogen
- syngas
- methane
- ethylene

## Code

Results can be reproduced using scripts contained in the `scripts/` directory. The chronology of running these scripts is as follows:

1. `QoIAwareProjection-train`
2. `QoIAwareProjection-VarianceData`
3. `QoIAwareProjection-kernel-regression`

Scripts 1. and 2. can take a long to run. Script 2. is parallelized and it is highly recommended that `QoIAwareProjection-VarianceData.py` is run on multiple CPUs. We have completed our computations running this script on 64CPUs, where looping over 100 random seeds for a single dataset takes about 20 hours to complete.

Our open-source Python library [**PCAfold**](https://pcafold.readthedocs.io/en/latest/index.html) is required.

## Jupyter notebooks

Once the results are obtained using these scripts, the following Jupyter notebooks can be used to post-process results and generate figures:

### Reproducing Figure 3




### Reproducing Figure 4


