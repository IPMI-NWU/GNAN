## Graph-based Neighbor-Aware Network for Gaze-Supervised Medical Image Segmentation

This project contains the training and testing code for the paper, as well as the model weights trained according to our method.

## Method
![1.0](Figure/fig-method1.png)

## Model Weights
The performance of our method is expressed as the mean and standard deviation of three different seed runs.
| |  NCI-ISBI |  KvasirSEG |
| :-: | :-: | :-: |
| Ours | 80.33±0.24 | 79.32±0.39 |
|Highest Dice| 80.53| 79.69|

The download links for our model weights are as [Weights](https://pan.baidu.com/s/1hiUfYfmO3XsEAawPg6Kazg?pwd=6be3).

## Datasets
The gaze of GazeMedSeg dataset can be downloaded at [Gaze](https://drive.google.com/drive/folders/1-38bG_81OsGVCb_trI00GSqfB_shCUQG).\
The GazeMedSeg dataset can download at [GazeMedSeg](https://drive.google.com/drive/folders/1XjgQ27R8zT8ymOTXohgl8HXntPEUbIXj).

The Kvasir dataset can download at [Kvasir](https://datasets.simula.no/kvasir-seg/).\
The NCI-ISBI dataset can download at [NCI](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/).

## Results
![1.0](Figure/fig-result1.png)

## Ablation Study
| $L_{pce}$ | $L_{UC}$ | $L_{NAP}$ | $L_{GCD}$ | NCI-ISBI | KvasirSEG |
| :-: | :-: | :-: | :-: | :-: | :-: |
| &#x2714;  | | | | 77.49±0.67 | 76.42±0.59 |
| &#x2714;  | &#x2714; | | | 78.29±0.60 | 77.09±0.46 |
| &#x2714;  | | &#x2714; | | 78.53±0.64 | 78.03±0.42 |
| &#x2714;  | &#x2714; | &#x2714; | | 79.11±0.55 | 78.54±0.71 |
| &#x2714;  | | &#x2714; | &#x2714; | 79.40±0.55 | 78.62±0.56 |
| &#x2714;  | &#x2714; | &#x2714; | &#x2714; | **80.33**±0.24 | **79.32**±0.39 |

Ablation study to evaluate the contribution of each module in GNAN. 

| Data Augmentation | NCI-ISBI | KvasirSEG |
| :-: | :-: | :-: |
| w/o | 78.38±0.59 | 77.66±0.43 |
| Cutout | 78.91±0.56 | 77.72±0.61 |
| Brightness-Contrast | 79.37±0.40 | 78.48±0.43 |
| Noise | 79.54±0.47 | 78.60±0.56 |
| Brightness-Contrast & Noise | **80.33**±0.24 | **79.32**±0.39 |

Ablation study on image perturbation methods. 
Due to space constraints, this table serves as a supplement to the further experiments in the paper.

| $\tau_b$   ($\tau_f$ = 0.6) | NCI-ISBI | KvasirSEG |
| :-: | :-: | :-: |
| 0.2 | 78.20±0.34 | 78.32±0.68 |
| 0.3 | **80.33**±0.24 | **79.32**±0.39 |
| 0.4 | 79.66±0.26 | 77.92±0.33 |
| **$\tau_f$  ($\tau_b$ = 0.3)** | **NCI-ISBI** | **KvasirSEG** |
| 0.5 | 78.88±0.27 | 78.42±0.40 |
| 0.6 | **80.33**±0.24 | **79.32**±0.39 |
| 0.7 | 79.66±0.26 | 77.92±0.33 |

Ablation study on foreground thresholds $\tau_f$ and background thresholds $\tau_b$.

## Requirements
```
python == 3.8
torch == 1.12.0
numpy == 1.24.4
medpy == 0.5.1
nibabel == 5.2.1
pandas == 2.0.3
scikit-image == 0.21.0
```
