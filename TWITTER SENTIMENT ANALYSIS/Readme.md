Emotion Classification on Twitter Data
A comparative study of three text classification approaches on the dair-ai/emotion dataset (Twitter messages labeled with 6 emotions: joy, sadness, anger, fear, love, surprise).
📊 Results Summary





























ModelTest AccuracyValidation AccuracyNotesFCNN (TF-IDF + Dense layers)87.25%87.35%Fast baseline, mild overfittingBidirectional LSTM88.10%89.85%Strong sequential modelingDistilBERT (fine-tuned Transformer)92.40%93.85%Best performance – clear winner with excellent generalization
🏆 Key Findings

Pretrained Transformers dramatically outperform classical and RNN-based methods.
Fine-tuning DistilBERT achieves near state-of-the-art (~93%) with only 3 epochs.
All models show good generalization (test ≈ validation scores).
Class imbalance was handled effectively with class weights (critical for LSTM).

📁 Project Structure
text.
├── data/
│   ├── train.txt
│   ├── validation.txt
│   └── test.txt
├── training.ipynb                # Full experiment notebook (local)
├── Pretrained_model_GPU.ipynb    # DistilBERT fine-tuning on Colab GPU
└── README.md
🚀 How to Run
Option 1: Local (FCNN + BiLSTM)

Install dependencies:Bashpip install pandas nltk tensorflow scikit-learn
Run training.ipynb in Jupyter.

Option 2: DistilBERT (Recommended – Best Results)

Open Pretrained_model_GPU.ipynb in Google Colab
Enable GPU: Runtime → Change runtime type → GPU
Upload train.txt and test.txt
Run all cells (~10 minutes total)

🔮 Prediction Example (DistilBERT)
Pythonfrom transformers import pipeline

classifier = pipeline("text-classification", model="your_saved_distilbert_model")
classifier("I feel so happy today!")  # → joy
📝 What I Learned

Start with simple baselines, but always try a Hugging Face pretrained model for text tasks.
Transformers win on accuracy and generalization with minimal tuning.
Proper evaluation (held-out validation) prevents test overfitting.

For any new text classification project: fine-tune a small pretrained Transformer first.
