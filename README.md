# Fine-Tuning and Ensembling Vision Transformers (DINOv2, SwinV2, BEiT)

> **Tags:** `#VisionTransformers` `#DINOv2` `#SwinV2` `#BEiT` `#TransferLearning` `#DeepLearning` `#XGBoost` `#ImageClassification` `#FeatureExtraction` `#FeatureEnsembling` `#HuggingFace` `#SingleNotebook` `#GPU`

This repository contains a single Jupyter notebook that demonstrates a complete pipeline for:

✅ Fine-tuning three Vision Transformer models: **DINOv2**, **SwinV2**, and **BEiT**  
✅ Extracting deep features (embeddings) from each model  
✅ Concatenating the features for **ensemble representation**  
✅ Training a final **XGBoost classifier** on the combined features  
✅ Evaluating the model using multiple metrics

---

## 🗂️ File Overview

```text
.
├── ViT_Ensemble_Classifier.ipynb   # 📓 Main notebook with all stages
├── checkpoints/                    # 💾 (Optional) Directory for saved model checkpoints
├── features/                       # 📁 (Optional) Directory for saved feature .npy files
├── README.md                       # 📄 This file


## Install Dependencies
pip install torch torchvision transformers scikit-learn xgboost numpy pandas

✅ Make sure you have a CUDA-capable GPU for efficient training and inference.

✅ 1. Dataset Preparation
Tags: #Dataset #HuggingFaceDatasets #DataLoader

Assumes a Hugging Face datasets.Dataset-like format.

Required fields:

'image' (as tensor)

'label' or label identifier in metadata

✅ 2. Fine-Tuning Vision Transformer Models
Tags: #ModelTraining #TransferLearning #DINOv2 #SwinV2 #BEiT

Fine-tunes each model using:

🧠 facebook/dinov2-base

🌀 microsoft/swinv2-base-patch4-window12-192-22k

🧩 microsoft/beit-base-patch16-224-pt22k-ft22k

from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    checkpoint_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
).to("cuda")


✅ 3. Feature Extraction
Tags: #FeatureExtraction #Embeddings #NoClassificationHead

Classification heads removed:
nn.Sequential(*list(model.children())[:-1])

Extracts pooler_output using a custom extract_features() function

Features are saved as .npy files for reuse

✅ 4. Feature Concatenation (Ensembling)
Tags: #Ensembling #MultiModelFusion #Embeddings

Concatenates deep features from all models:

python
Copy
Edit
combined_features = torch.concat([
    dino_features,
    swin_features,
    beit_features
], dim=1)
These combined embeddings are input into the final classifier.



✅ 5. Final Classification with XGBoost
Tags: #XGBoost #Classifier #EnsembleModel

Trains a gradient boosting classifier using the combined features:

python
Copy
Edit
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    tree_method='gpu_hist',
    objective='multi:softmax',
    num_class=4,  # Update based on dataset
    max_depth=4,
    learning_rate=0.1,
    n_estimators=1000,
    subsample=0.7
)

xgb_model.fit(X_train, y_train)

✅ 6. Evaluation Metrics
Tags: #Metrics #ModelEvaluation #ClassificationMetrics

Evaluates performance with scikit-learn metrics:

python
Copy
Edit
from sklearn import metrics

print("Accuracy      :", metrics.accuracy_score(y_test, y_pred))
print("Precision     :", metrics.precision_score(y_test, y_pred, average='weighted'))
print("Recall        :", metrics.recall_score(y_test, y_pred, average='weighted'))
print("F1 Score      :", metrics.f1_score(y_test, y_pred, average='weighted'))
print("Cohen Kappa   :", metrics.cohen_kappa_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred, target_names=target_names))

