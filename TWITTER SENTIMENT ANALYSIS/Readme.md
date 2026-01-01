# Twitter Emotion Classification Project

A comparative study of text classification models on the **dair-ai/emotion** dataset (Twitter messages labeled with 6 emotions: joy, sadness, anger, fear, love, surprise).

## 📊 Final Results

| Model                              | Test Accuracy | Validation Accuracy | Notes                              |
|------------------------------------|---------------|----------------------|------------------------------------|
| FCNN (TF-IDF + Dense)              | 87.25%        | 87.35%              | Fast baseline, mild overfitting    |
| Bidirectional LSTM                 | 88.10%        | 89.85%              | Strong, excellent generalization   |
| **DistilBERT (fine-tuned Transformer)** | **92.40%**    | **93.85%**          | **Best model** – clear winner      |

## 🏆 Key Takeaways
- Pretrained Transformers significantly outperform classical and RNN approaches.
- DistilBERT achieves near state-of-the-art (~93%) with only 3 epochs.
- All models generalize well (test ≈ validation scores).
- Class imbalance was effectively handled with class weights (critical for LSTM).


## 🚀 How to Run

### 1. Local Models (FCNN + BiLSTM)
```bash
pip install pandas nltk tensorflow scikit-learn matplotlib

Open and run *.ipynb in Jupyter

2. Best Model – DistilBERT (Recommended)

Open Pretrained_model_GPU.ipynb in Google Colab
Enable GPU: Runtime → Change runtime type → GPU
Upload train.txt and test.txt
Run all cells (~10 minutes)


🔮 Quick Prediction (DistilBERT)

from transformers import pipeline

classifier = pipeline("text-classification", model="./your_saved_distilbert_model")
print(classifier("I feel so happy today!"))  # → joy
