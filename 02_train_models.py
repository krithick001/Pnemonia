"""
Step 2: Train and compare three models on the clinical notes.
  A) TF-IDF + Logistic Regression   (NLP baseline)
  B) Bidirectional LSTM             (sequential model)  <- main model
  C) Random Forest on tabular data  (reference, uses original columns)
Saves metrics, plots and the trained models into outputs/.
"""
import json, re, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve,
                             precision_score, recall_score, accuracy_score)

SEED = 42
np.random.seed(SEED)

# ---------- data ----------
notes = pd.read_csv("data/clinical_notes.csv")
notes["text"] = notes["text"].astype(str).str.lower().apply(
    lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", t)).strip())

idx = np.arange(len(notes))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=notes["label"], random_state=SEED)
X_train, X_test = notes["text"].iloc[tr_idx], notes["text"].iloc[te_idx]
y_train, y_test = notes["label"].iloc[tr_idx].values, notes["label"].iloc[te_idx].values

results = {}


def score(name, y_true, prob):
    pred = (prob > 0.5).astype(int)
    results[name] = {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "roc_auc": round(roc_auc_score(y_true, prob), 4),
    }
    print(f"\n===== {name} =====")
    print(classification_report(y_true, pred, target_names=["No pneumonia", "Pneumonia"]))
    print("ROC-AUC:", results[name]["roc_auc"])
    return pred


curves = {}

# ---------- A) TF-IDF + Logistic Regression ----------
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(tfidf.fit_transform(X_train), y_train)
p_lr = lr.predict_proba(tfidf.transform(X_test))[:, 1]
score("TF-IDF + Logistic Regression", y_test, p_lr)
curves["TF-IDF + LogReg"] = p_lr
pickle.dump((tfidf, lr), open("outputs/tfidf_logreg.pkl", "wb"))

# ---------- C) Random Forest on original tabular columns ----------
raw = pd.read_csv("data/pneumonia_dataset.csv", keep_default_na=False).replace("-", "")
raw = raw.drop(columns=["PatientID", "Xray", "WBC_count"])
raw["Oxygen_saturation"] = pd.to_numeric(raw["Oxygen_saturation"], errors="coerce")
raw["Oxygen_saturation"] = raw["Oxygen_saturation"].fillna(raw["Oxygen_saturation"].median())
y_raw = (raw.pop("Diagnosis") == "Yes").astype(int).values
Xtab = pd.get_dummies(raw)
rf = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced")
rf.fit(Xtab.iloc[tr_idx], y_raw[tr_idx])
p_rf = rf.predict_proba(Xtab.iloc[te_idx])[:, 1]
score("Random Forest (tabular)", y_raw[te_idx], p_rf)
curves["Random Forest (tabular)"] = p_rf

# ---------- B) BiLSTM ----------
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(SEED)
    MAX_WORDS, MAX_LEN = 3000, 60

    tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(X_train)
    Xtr = pad_sequences(tok.texts_to_sequences(X_train), maxlen=MAX_LEN)
    Xte = pad_sequences(tok.texts_to_sequences(X_test), maxlen=MAX_LEN)

    model = Sequential([
        Embedding(MAX_WORDS, 64),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    hist = model.fit(Xtr, y_train, epochs=30, batch_size=32, validation_split=0.15,
                     callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                     verbose=2)
    p_lstm = model.predict(Xte).ravel()
    score("BiLSTM", y_test, p_lstm)
    curves["BiLSTM"] = p_lstm

    model.save("outputs/bilstm_model.keras")
    pickle.dump(tok, open("outputs/tokenizer.pkl", "wb"))

    plt.figure(figsize=(9, 3.5))
    plt.subplot(1, 2, 1); plt.plot(hist.history["loss"], label="train"); plt.plot(hist.history["val_loss"], label="val")
    plt.title("Loss"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(hist.history["accuracy"], label="train"); plt.plot(hist.history["val_accuracy"], label="val")
    plt.title("Accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig("outputs/bilstm_training_curves.png", dpi=150); plt.close()
except ImportError:
    print("\nTensorFlow not installed -> skipping BiLSTM. Run: pip install tensorflow")

# ---------- results ----------
pd.DataFrame(results).T.to_csv("outputs/model_comparison.csv")
json.dump(results, open("outputs/metrics.json", "w"), indent=2)
print("\n\n=== MODEL COMPARISON ===")
print(pd.DataFrame(results).T)

# ROC curves
plt.figure(figsize=(5.5, 5))
for name, p in curves.items():
    yt = y_raw[te_idx] if "tabular" in name else y_test
    fpr, tpr, _ = roc_curve(yt, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yt, p):.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=150); plt.close()

# Confusion matrix of best text model
best = max([k for k in curves if "tabular" not in k], key=lambda k: roc_auc_score(y_test, curves[k]))
ConfusionMatrixDisplay(confusion_matrix(y_test, curves[best] > 0.5),
                       display_labels=["No", "Pneumonia"]).plot(cmap="Blues")
plt.title(f"Confusion matrix: {best}"); plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150); plt.close()
print("\nPlots and models saved in outputs/")
