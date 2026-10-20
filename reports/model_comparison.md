# Phase 5 Model Comparison

Models are ranked using validation macro-F1, with weighted-F1, accuracy,
and validation loss as successive tie-breakers.

| Rank | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Loss |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | efficientnet_b0 | 0.1094 | 0.0320 | 0.1094 | 0.0474 | 0.0474 | 4.0638 |
| 2 | resnet50 | 0.0469 | 0.0207 | 0.0469 | 0.0194 | 0.0194 | 4.0826 |
| 3 | vit_b_16 | 0.0156 | 0.0095 | 0.0156 | 0.0107 | 0.0107 | 4.2351 |
| 4 | baseline_cnn | 0.0312 | 0.0041 | 0.0312 | 0.0064 | 0.0064 | 4.8331 |

Selected production model: **efficientnet_b0**
