# Wavelet-Guided Frequency Decoupling for Channel-Robust UAV RFFI

[![Status](https://img.shields.io/badge/Status-Submitted%20to%20IEEE%20WCL-orange.svg)](#publication-status)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation accompanying the manuscript:

> **Wavelet-Guided Frequency Decoupling for Channel-Robust UAV RFFI**  
> Zhenxin Cai, Yu Wang, and Jin Sha  
> Manuscript submitted to *IEEE Wireless Communications Letters*.

## Publication Status

> [!IMPORTANT]
> This manuscript has been submitted to **IEEE Wireless Communications Letters** and has **not yet been formally published**. The title, method description, experimental results, and citation information may be updated during the review process.

## Overview

Radio frequency fingerprint identification (RFFI) identifies wireless devices through transmitter-specific hardware imperfections, such as I/Q imbalance, power-amplifier nonlinearity, and oscillator phase noise. For unmanned aerial vehicle (UAV) identification, a major practical challenge is the distribution shift between controlled line-of-sight (LOS) acquisition and complex non-line-of-sight (NLOS) deployment.

This project formulates **LOS-to-NLOS UAV RFFI as a domain generalization problem**: only LOS samples are available during training, whereas NLOS samples are reserved exclusively for testing. To improve channel robustness, we propose a **Wavelet-Guided Frequency Decoupling (WGFD)** framework that performs multilevel 2D discrete wavelet transform on task-supervised intermediate feature maps.


- approximation responses are emphasized because they are comparatively less sensitive to rapid NLOS-induced local variations;
- selected detail responses are retained because they may contain complementary device-specific fingerprint cues;
- spatial and wavelet-domain representations are jointly fused rather than treating wavelet subbands as isolated inputs.

> [!NOTE]
> In this project, “low frequency” and “high frequency” refer to **wavelet-domain spatial frequencies over the time-frequency plane**, rather than the absolute RF carrier frequency.


## Framework

<!-- Export Fig. 1 of the manuscript as assets/framework.png before enabling this block. -->

<!--
<p align="center">
  <img src="assets/framework.png" width="100%" alt="WGFD framework">
</p>
-->

The overall pipeline is:

```text
Logarithmic spectrogram
        │
        ▼
PConv-based direction-aware stem
        │
        ▼
CSPA backbone × 3
  └── stacked MSFM blocks
       ├── identity branch
       ├── local spatial branch
       └── WGFD branch
            ├── Level-1 2D-DWT
            ├── Level-2 2D-DWT
            ├── selected subband fusion
            └── channel attention
        │
        ▼
1 × 1 convolution → global average pooling → dropout → linear classifier
        │
        ▼
18 UAV individual classes
```

## Repository Structure

```text
Channel-Robust-UAV-RFFI/
├── WGFD.py              # Model definition: PConv, WGFD, MSFM, CSPA, and Extractor
├── train_signal.py      # Training, validation, checkpointing, and NLOS testing
├── dataloader.py        # Data loading and preprocessing required by train_signal.py
└── README.md
```

## Dataset

Experiments are conducted on the **DroneRFb-DIR** dataset.

- Dataset page: https://www.scidb.cn/en/detail?dataSetId=84cf9101e739402784b1396783881202
- Dataset paper: https://doi.org/10.11999/JEIT240804

The experimental subset used in the manuscript contains:

| Item | Description |
|---|---|
| UAV categories | 6 |
| Devices per category | 3 |
| Total identities | 18 |
| Propagation conditions | LOS and NLOS |
| Training/validation data | LOS only |
| Test data | NLOS only |
| LOS split | 80% training / 20% validation |
| Model input | Logarithmic spectrogram |
| Input tensor shape | `N × 1 × 350 × 2000` |

The NLOS condition contains building blockage between the transmitter and receiver. No NLOS samples are used for model training in the reported LOS→NLOS evaluation.

### Default Script Configuration

| Parameter | Default value |
|---|---:|
| Training batch size | 16 |
| Validation batch size | 8 |
| Epochs | 100 |
| Learning rate | `1e-3` |
| Optimizer | Adam |
| Weight decay | 0 |
| Loss | NLLLoss |
| Random seed | 300 |

The submitted manuscript reports experiments conducted in PyTorch on an NVIDIA RTX 3090 GPU, using a learning rate of `0.001` for 100 epochs.

## Experimental Protocol

Two protocols are reported:

| Protocol | Training data | Test data | Purpose |
|---|---|---|---|
| LOS→LOS | LOS | LOS | In-domain identification |
| LOS→NLOS | LOS | NLOS | Cross-channel generalization |

The LOS→NLOS protocol is the primary evaluation setting. NLOS data must remain unavailable during training and validation.

## Main Results

Performance comparison on DroneRFb-DIR:

| Method | LOS→LOS Acc. (%) | LOS→LOS F1 (%) | LOS→NLOS Acc. (%) | LOS→NLOS F1 (%) |
|---|---:|---:|---:|---:|
| ROA | 99.46 | 99.46 | 84.33 | 85.67 |
| SigMix | 89.11 | 89.42 | 53.76 | 50.25 |
| DACL | 91.48 | 91.81 | 73.66 | 67.59 |
| C2C-SVM | 98.96 | 97.69 | 69.40 | 71.80 |
| CIS | 99.28 | 98.82 | 66.25 | 67.06 |
| DoLoS | **100.00** | **100.00** | 71.09 | 73.43 |
| **WGFD (proposed)** | **100.00** | **100.00** | **95.27** | **94.84** |

WGFD improves LOS→NLOS accuracy by **10.94 percentage points** over the strongest comparison method, ROA, without using NLOS training data.

In the noise-robustness experiment, WGFD achieves **78.49% accuracy at −5 dB**, and its accuracy approaches **99% at high SNRs**.

## Ablation Studies

### Contribution of the Main Components

| Variant | LOS→LOS Acc. (%) | LOS→LOS F1 (%) | LOS→NLOS Acc. (%) | LOS→NLOS F1 (%) |
|---|---:|---:|---:|---:|
| WGFD without PConv | 99.46 | 99.46 | 92.11 | 91.99 |
| WGFD without CSPA | 98.39 | 98.41 | 85.28 | 85.54 |
| WGFD without MSFM | 100.00 | 100.00 | 88.93 | 88.92 |
| **Full WGFD** | **100.00** | **100.00** | **95.27** | **94.84** |

### Importance of WGFD Subbands

| `z_LL1` | `z_LL2` | `z_H1` | `z_H2` | LOS→NLOS Acc. (%) | LOS→NLOS F1 (%) |
|:---:|:---:|:---:|:---:|---:|---:|
| ✓ | ✓ | ✓ | ✓ | 93.06 | 93.01 |
| × | × | ✓ | ✓ | 85.17 | 86.46 |
| ✓ | ✓ | × | × | 88.96 | 88.86 |
| × | ✓ | ✓ | ✓ | 90.85 | 90.45 |
| ✓ | × | ✓ | ✓ | 86.12 | 86.80 |
| ✓ | ✓ | × | ✓ | 91.79 | 92.03 |
| **✓** | **✓** | **✓** | **×** | **95.27** | **94.84** |

Using only approximation branches outperforms using only detail branches under LOS→NLOS (`88.96%` versus `85.17%`). The best result is obtained by retaining `z_LL1`, `z_LL2`, and `z_H1`, while removing `z_H2`.

### Importance of MSFM Branches

| `z_spa` | `z_local` | `z_freq` | LOS→NLOS Acc. (%) | LOS→NLOS F1 (%) |
|:---:|:---:|:---:|---:|---:|
| × | ✓ | ✓ | 90.85 | 90.47 |
| ✓ | × | ✓ | 75.08 | 74.92 |
| ✓ | ✓ | × | 68.77 | 66.78 |
| **✓** | **✓** | **✓** | **95.27** | **94.84** |

Removing the frequency-decoupled branch causes the largest degradation, while the complete three-stream design gives the best result.

## Citation

Because the manuscript has not yet been published, please use the following temporary citation:

```bibtex
@article{cai2026wgfd,
  title  = {Wavelet-Guided Frequency Decoupling for Channel-Robust UAV RFFI},
  author = {Cai, Zhenxin and Wang, Yu and Sha, Jin},
  note   = {Manuscript submitted to IEEE Wireless Communications Letters},
  year   = {2026}
}
```

The citation will be updated after the final publication information becomes available.

When using DroneRFb-DIR, please also cite the dataset paper:

```bibtex
@article{ren2025dronerfb,
  title   = {DroneRFb-DIR: An RF Signal Dataset for Non-cooperative Drone Individual Identification},
  author  = {Ren, Junyu and Yu, Ningning and Zhou, Chengwei and Shi, Zhiguo and Chen, Jiming},
  journal = {Journal of Electronics and Information Technology},
  volume  = {47},
  number  = {3},
  pages   = {573--581},
  year    = {2025},
  doi     = {10.11999/JEIT240804}
}
```

## Acknowledgements

We gratefully acknowledge the authors of DroneRFb-DIR for releasing the UAV RF signal dataset. 

## Contact

For questions about the manuscript or code, please contact:

- Zhenxin Cai: `652022230002@smail.nju.edu.cn`

