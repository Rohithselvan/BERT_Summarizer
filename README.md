# 🧠 BERT Summarizer – GPU Fine-Tuned Model

An intelligent **text summarization system** built using a fine-tuned **BERT Encoder–Decoder model**.  
This project performs **abstractive summarization** on long text passages, trained on a **custom NLP dataset**.  

🪄 **Key Highlights**
- Fine-tuned **BERT2BERT** model using PyTorch and Hugging Face Transformers  
- Datasets hosted on **Kaggle**  
- Final trained model hosted on **Hugging Face** for instant use  
- Supports GPU-accelerated inference for faster results  

📍 **Resources**
- **Model:** [Rohith1872/bert-summarizer-gpu-final](https://huggingface.co/Rohith1872/bert-summarizer-gpu-final)  
- **Raw Dataset:** [Kaggle – NLP Project](https://www.kaggle.com/datasets/yoo1234567/nlp-proj)  
- **Processed Dataset:** [Kaggle – Processed Dataset](https://www.kaggle.com/datasets/yoo1234567/dataset)

---

> 💡 *Clone the repo, open the notebooks in Kaggle, or directly load the Hugging Face model to start summarizing your own text.*

---

## 📘 Overview

This project demonstrates a **BERT Encoder–Decoder** model fine-tuned for **text summarization**.  
It uses `bert-base-uncased` as both encoder and decoder, fine-tuned on a custom dataset in a Kaggle GPU environment.  
The model is now available for public inference on Hugging Face.

---

## 📂 Project Structure
```
📁 BERT-Summarizer/
├── 📓 1_dataset_preprocessing.ipynb
├── 📓 2_model_training.ipynb
├── 📓 3_model_testing.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```


| File | Description |
|------|--------------|
| **1_dataset_preprocessing.ipynb** | Cleans and preprocesses the raw dataset. |
| **2_model_training.ipynb** | Fine-tunes the BERT encoder–decoder model. |
| **3_model_testing.ipynb** | Loads the model from Hugging Face and generates summaries. |

---

## 📊 Datasets

### 🔹 Raw Dataset
Base data used for model training.  
👉 [**NLP Project Raw Dataset**](https://www.kaggle.com/datasets/yoo1234567/nlp-proj)

### 🔹 Processed Dataset
Cleaned, tokenized, and ready for model input.  
👉 [**Processed Dataset**](https://www.kaggle.com/datasets/yoo1234567/dataset)

---

## 🤖 Model Information

The fine-tuned summarization model is hosted on **Hugging Face**:

👉 [**Rohith1872/bert-summarizer-gpu-final**](https://huggingface.co/Rohith1872/bert-summarizer-gpu-final)

### 🔸 Model Architecture
- **Base Model:** `bert-base-uncased`  
- **Type:** Encoder–Decoder (BERT2BERT)  
- **Training Platform:** Kaggle GPU (Tesla T4)  
- **Framework:** PyTorch + Hugging Face Transformers  

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧩 Training Workflow
Preprocessing:
Load and clean the raw dataset (remove stopwords, normalize text, prepare input-output pairs).

Training:
Fine-tune BERT using the processed dataset for abstractive summarization.

Testing:
Evaluate the model and generate human-readable summaries.

Deployment:
Upload final model and tokenizer to Hugging Face using push_to_hub().

## 📈 Results
Metric	Description	Score
ROUGE-1	Overlap of unigrams	High (Good coherence)
ROUGE-L	Longest common subsequence	Stable performance
Inference Speed	~1.2 s per paragraph (on GPU)	

(Exact metrics depend on dataset split and training parameters.)

## 🧠 Key Features
Custom fine-tuned BERT2BERT model for abstractive summarization

Preprocessed dataset for reproducible training

GPU-optimized for fast experimentation

Public Hugging Face model for open use

## 🧰 Environment
Platform: Kaggle

Runtime: GPU (Tesla T4)

Language: Python 3.11

Framework: PyTorch, Hugging Face Transformers

## 👤 Author
Rohith Selvan
📧 rohithselvan10@gmail.com
🌐 Hugging Face Profile

## 🏁 License
Feel free to use, modify, and share with attribution.

## 📜 Citation
@misc{rohith1872_bert_summarizer_2025,
  author = {Rohith Selvan},
  title  = {BERT Summarizer (GPU Fine-Tuned)},
  year   = {2025},
  howpublished = {\url{https://huggingface.co/Rohith1872/bert-summarizer-gpu-final}}
}
