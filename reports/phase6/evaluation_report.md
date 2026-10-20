# Phase 6 Model Evaluation

- Model: `efficientnet_b0`
- Checkpoint: `models\best_model.pth`
- Split: `test`
- Samples: 128
- Loss: 4.0514
- Top-1 accuracy: 8.59%
- Top-5 accuracy: 25.00%
- Macro F1: 5.28%
- Weighted F1: 5.28%
- Mean confidence: 2.91%
- Expected calibration error: 5.68%
- Errors: 117 (91.41%)

## Ten Lowest-Recall Classes

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Alstonia_Scholaris___diseased | 0.00% | 0.00% | 0.00% | 2 |
| Alstonia_Scholaris___healthy | 0.00% | 0.00% | 0.00% | 2 |
| Apple___Apple_scab | 0.00% | 0.00% | 0.00% | 2 |
| Apple___Black_rot | 0.00% | 0.00% | 0.00% | 2 |
| Apple___Cedar_apple_rust | 0.00% | 0.00% | 0.00% | 2 |
| Apple___healthy | 0.00% | 0.00% | 0.00% | 2 |
| Arjun___healthy | 0.00% | 0.00% | 0.00% | 2 |
| Basil___healthy_(P8) | 0.00% | 0.00% | 0.00% | 2 |
| Cherry_(including_sour)___Powdery_mildew | 0.00% | 0.00% | 0.00% | 2 |
| Cherry_(including_sour)___healthy | 0.00% | 0.00% | 0.00% | 2 |

## Most Frequent Confusions

| True class | Predicted class | Count |
|---|---|---:|
| Pepper__bell___healthy | Apple___healthy | 2 |
| Chinar___diseased | Chinar___healthy | 2 |
| Peach___healthy | Apple___healthy | 2 |
| Soybean___healthy | Apple___Black_rot | 2 |
| Pepper,_bell___healthy | Apple___Black_rot | 2 |
| Peach___Bacterial_spot | Apple___healthy | 2 |
| Cherry_(including_sour)___Powdery_mildew | Apple___Black_rot | 2 |
| Arjun___healthy | Pongamia_Pinnata___healthy | 2 |
| Tomato___Tomato_mosaic_virus | Arjun___diseased | 2 |
| Tomato___Spider_mites Two-spotted_spider_mite | Pepper__bell___healthy | 1 |

> Results must be interpreted using the training and evaluation scope 
> documented in the project README.
