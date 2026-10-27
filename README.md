# Multi-Crop Leaf Disease Identification using Improved Deep CNNs

PyTorch re-implementation and extension of *"Identification of Maize Leaf Diseases Using Improved Deep Convolutional Neural Networks"* (Zhang et al., IEEE Access, 2018, [DOI: 10.1109/ACCESS.2018.2844405](https://doi.org/10.1109/ACCESS.2018.2844405)) to **four crops and 18 classes**: Tomato, Potato, Bell Pepper and Rice.

Minor Project (B.Tech, Computer Engineering) — Silver Oak University, Ahmedabad, October 2026.
Authors: Rishabh Soni, Jennis Joshi, Virti Shah. Guide: Mr. Sanket Shah.

> The vision engine here is the foundation for a planned **multilingual (Gujarati / Hindi / English) farmer advisory app**. Farmers photograph a leaf and get a diagnosis plus treatment advice in their own language.

---

## Why this project

The base paper proposed lightweight GoogLeNet and 3-layer Cifar10-style CNN modifications, but only tested them on maize and trained in Caffe with plain SGD. We tested whether the claims generalise to other crops, and fixed several weaknesses:

| Component | Base paper (Zhang et al., 2018) | This project |
|---|---|---|
| Crops | Maize (9 classes) | Tomato (10), Potato (3), Bell Pepper (2), Rice (3) = 18 classes |
| Framework | Caffe | PyTorch |
| Optimizer | SGD (momentum 0.9), 100k iterations | AdamW (lr 1e-3, weight decay 1e-4) + Cosine Annealing |
| Loss | Cross-entropy | Cross-entropy with 0.1 label smoothing |
| Train/test split | Random split after augmentation (leakage risk) | **Leakage-safe split by original image** (`origin_key`), with assertion |
| Feature-map bridge | Fixed flatten | `AdaptiveAvgPool2d((6, 6))` to avoid parameter explosion at 224x224 |

## Key contributions

1. **9x offline augmentation (OpenCV):** original, 90/180/270 degree rotations, horizontal flip of each of the three rotations, 80% center crop, and 3-channel grayscale.
2. **Leakage-safe splitting:** a regex-based `origin_key` groups all 9 variants of a leaf so they land together in either the 80% train or the 20% test split. This is checked with `assert _train_origins.isdisjoint(_val_origins)` before training.
3. **Three models compared:** fine-tuned GoogLeNet, a 4-layer CifarStyleCNN, and `PaperCifarNet` (a 3-layer reproduction of the paper's Fig. 8 with configurable pooling and dropout).
4. **Two ablation studies:** dropout 0.50-0.75 (six values), then four pooling combinations (Max-Ave-Ave, Max-Max-Ave, Max-Max-Max, Ave-Ave-Ave).

## Datasets

| Crop | Source | Classes | Originals | After 9x augmentation | Train (80%) | Test (20%) |
|---|---|---|---|---|---|---|
| Tomato | PlantVillage | 10 | 501 | 4,509 | 3,608 | 901 |
| Potato | PlantVillage | 3 (Early blight, Late blight, Healthy) | 450 | 4,050 | 3,240 | 810 |
| Bell Pepper | PlantVillage | 2 (Bacterial spot, Healthy) | 300 | 2,700 | 2,160 | 540 |
| Rice | Kaggle [`vbookshelf/rice-leaf-diseases`](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases) | 3 (Bacterial leaf blight, Brown spot, Leaf smut) | 120 | 1,080 | 864 | 216 |

PlantVillage images are studio-lit with clean backgrounds. The Rice images are field-captured with noisy backgrounds.

**The datasets are not included in this repository** (size and licensing). Download them from Kaggle and place them as described under [Usage](#usage). The Rice dataset has no healthy class.

## Results

### Baseline: GoogLeNet vs 4-layer CifarStyleCNN (best validation accuracy)

| Crop | Classes | GoogLeNet (fine-tuned) | 4-layer CifarStyleCNN |
|---|---|---|---|
| Tomato | 10 | 99.89% | 93.67% |
| Potato | 3 | 99.88% | 98.77% |
| Bell Pepper | 2 | 99.81% | 98.70% |
| Rice | 3 | 92.13% | 92.13% |

### Ablation 1: dropout sweep on PaperCifarNet (Max-Max-Ave pooling), test accuracy %

| Dropout | Tomato | Potato | Bell Pepper | Rice |
|---|---|---|---|---|
| 0.50 | 90.3 | 96.4 | 97.6 | 72.7 |
| 0.55 | **90.9** | 96.9 | 97.6 | 74.5 |
| 0.60 | 85.0 | 94.3 | 95.0 | **81.9** |
| 0.65 | 87.3 | 96.0 | 94.6 | 76.9 |
| 0.70 | 87.8 | 94.9 | 96.3 | 77.8 |
| 0.75 | 84.7 | **97.9** | **97.8** | 65.3 |

### Ablation 2: pooling combination sweep on PaperCifarNet, test accuracy %

| Pooling | Tomato | Potato | Bell Pepper | Rice |
|---|---|---|---|---|
| Max-Ave-Ave | 89.8 | 95.2 | 98.0 | 90.3 |
| Max-Max-Ave | 91.8 | 95.9 | 98.3 | 89.8 |
| Max-Max-Max | **94.3** | **97.8** | **98.9** | 87.5 |
| Ave-Ave-Ave | 93.9 | 95.4 | 97.4 | **92.1** |

### Main findings

- Fine-tuned **GoogLeNet** reaches about 99.8-99.9% on the three studio-lit PlantVillage crops.
- The compact **PaperCifarNet** reaches up to **98.9%** (Bell Pepper), a small footprint suited to mobile/edge use.
- **Studio leaves (Tomato, Potato, Bell Pepper):** Max-Max-Max pooling works best because it keeps sharp lesion edges.
- **Field leaves (Rice):** Ave-Ave-Ave pooling works best (92.1%) because it smooths background soil and lighting noise.
- The base paper's Max-Max-Ave choice did not carry over to the other crops.

## Models

| Model | Description |
|---|---|
| GoogLeNet | ImageNet-pretrained Inception v1, classifier head resized per crop, two auxiliary heads (loss weight 0.3 each) during training |
| CifarStyleCNN | 4 conv blocks (3 > 32 > 64 > 128 > 128), alternating max/avg pooling, adaptive pool, dropout head |
| PaperCifarNet | Conv(3>32, 5x5) > Conv(32>32, 5x5) > Conv(32>64, 5x5), configurable pooling per layer, `AdaptiveAvgPool2d((6,6))`, FC 2304 > 256 > classes, configurable dropout |

## Training recipe (shared by all models)

| Setting | Value |
|---|---|
| Input size | 224 x 224, ImageNet mean/std normalisation |
| Optimizer | AdamW, lr = 1e-3, weight decay = 1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss | Cross-entropy, label smoothing 0.1 |
| Batch size | 32 (64 on larger GPUs) |
| Split | 80/20 by original image (leakage-safe) |

Hardware: NVIDIA RTX 3050 (6 GB) for local Tomato runs; Kaggle Tesla T4 for Potato, Bell Pepper and Rice sweeps.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── notebooks/
│   ├── tomato_training.ipynb
│   ├── potato_training.ipynb
│   ├── bell_pepper_training.ipynb
│   └── rice_training.ipynb
├── docs/
│   └── REPORT.pdf            # full project report
└── results/                  # optional: training-history CSVs and curve images
```

Each notebook is self-contained: dataset check, leakage-safe split, baseline training (GoogLeNet and CifarStyleCNN), then the dropout and pooling ablations. Training saves the best checkpoint (`.pth`) and an epoch-by-epoch history (`.csv`).

## Usage

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   pip install -r requirements.txt
   ```
2. Download the datasets (PlantVillage for Tomato/Potato/Bell Pepper, Kaggle Rice) and run the 9x augmentation so each crop has one folder per class.
3. Open the notebook for your crop (in Kaggle or locally with a CUDA GPU) and set `DATA_DIR` in the config cell to your dataset path.
4. Run all cells. Turn **Internet on** in Kaggle once so the pretrained GoogLeNet weights can download.

If you hit an out-of-memory error, reduce `BATCH_SIZE` to 16.

## Limitations

- The Rice dataset contains **no healthy class**.
- The models classify **single, isolated leaves**; they are not object detectors for whole-field or canopy images.
- Only **visible foliar symptoms** are detected (not root rot or soil issues without leaf discoloration).
- PlantVillage is studio-captured; accuracy on real field photos of those crops has not been measured.

## Future work

- Export to ONNX / TensorFlow Lite and build the multilingual (Gujarati / Hindi / English) mobile app with treatment advice.
- Collect healthy field-captured rice leaves in Gujarat.
- Move to object detection (YOLOv8 / YOLOv11) for multiple diseased leaves per image.

## Reference

X. Zhang, Y. Qiao, F. Meng, C. Fan and M. Zhang, "Identification of Maize Leaf Diseases Using Improved Deep Convolutional Neural Networks," *IEEE Access*, vol. 6, pp. 30370-30377, 2018. DOI: 10.1109/ACCESS.2018.2844405

Other works cited in the report include Mohanty et al. (2016) and Hughes and Salathe (2015) for PlantVillage; see `docs/REPORT.pdf` for the full list.

## License

Released under the MIT License (see `LICENSE`). Datasets keep their own licenses.
