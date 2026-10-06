# Automated Pneumonia Detection using NLP and Sequential Modeling

Classifies a patient's clinical note as **pneumonia / no pneumonia** using NLP
(TF-IDF) and a sequential deep-learning model (Bidirectional LSTM).

## Project structure
```
pneumonia_project/
├── data/
│   ├── pneumonia_dataset.csv   # original dataset (710 patients, structured)
│   └── clinical_notes.csv      # generated text notes + labels (step 1 output)
├── 01_prepare_data.py          # clean CSV -> clinical text notes
├── 02_train_models.py          # train + compare 3 models, save plots/metrics
├── 03_predict.py               # predict on a new note
├── outputs/                    # models, metrics, plots (created by step 2)
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python 01_prepare_data.py
python 02_train_models.py
python 03_predict.py "your clinical note here"
```
(Google Colab works too: upload the folder and run the same commands with `!`.)

## Method
1. **Data cleaning:** `-` = missing; the word `None` is kept as a valid value ("no symptom").
   `Xray` and `WBC_count` (99% empty) and `PatientID` are dropped.
2. **Text generation:** each row becomes a short note, e.g.
   *"3 year old male patient. presents with bloody cough. high fever reported. ... crackles heard on auscultation."*
   Absent symptoms are written as negations ("no fever", "denies shortness of breath"),
   which is the main thing a sequence model must handle.
3. **Preprocessing:** lowercase, remove punctuation, tokenize, pad to 60 tokens.
4. **Models:**
   - TF-IDF (1-2 grams) + Logistic Regression  (NLP baseline)
   - Embedding -> Bidirectional LSTM -> Dropout -> Dense (sigmoid)  (sequential model)
   - Random Forest on the original columns  (reference)
5. **Evaluation:** 80/20 stratified split; accuracy, precision, recall, F1, ROC-AUC,
   confusion matrix, ROC curves.

## Results (tested)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.958 | 1.000 | 0.927 | 0.962 | 0.995 |
| Random Forest (tabular) | 0.986 | 1.000 | 0.976 | 0.988 | 0.995 |
| BiLSTM | run `02_train_models.py` with TensorFlow installed | | | | |

## Limitations (mention these in your report)
- The notes are **generated from structured columns**, not written by clinicians, so
  the NLP task is simpler than real radiology/clinical text.
- The dataset is small (710 rows) and scores are very high, which suggests the labels
  are strongly tied to the symptom columns (possibly synthetic data).
  Results will not transfer to real hospital data.
- Educational project only; not for clinical use.

## Future work
Use real clinical text (Indiana University CXR / MIMIC-CXR reports), pretrained
embeddings (BioWordVec) or ClinicalBERT, and add a Streamlit app.
