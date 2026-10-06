"""
Step 3: Predict pneumonia from a new clinical note.
Usage:  python 03_predict.py "62 year old male patient. wet cough. high fever. ..."
Uses the BiLSTM if available, otherwise the TF-IDF model.
"""
import sys, re, pickle, os

text = sys.argv[1] if len(sys.argv) > 1 else (
    "62 year old male patient. presents with wet cough. high fever reported. "
    "severe shortness of breath. moderate chest pain. mild fatigue. patient is confused. "
    "oxygen saturation 88 percent, low. crackles heard on auscultation. green sputum. "
    "temperature 39.5 degrees celsius.")

clean = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()

if os.path.exists("outputs/bilstm_model.keras"):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tok = pickle.load(open("outputs/tokenizer.pkl", "rb"))
    model = load_model("outputs/bilstm_model.keras")
    prob = float(model.predict(pad_sequences(tok.texts_to_sequences([clean]), maxlen=60), verbose=0)[0][0])
    used = "BiLSTM"
else:
    tfidf, lr = pickle.load(open("outputs/tfidf_logreg.pkl", "rb"))
    prob = float(lr.predict_proba(tfidf.transform([clean]))[0][1])
    used = "TF-IDF + Logistic Regression"

print(f"Model: {used}")
print(f"Pneumonia probability: {prob:.2%}")
print("Prediction:", "PNEUMONIA" if prob > 0.5 else "NO PNEUMONIA")
print("(Educational project - not for real medical use.)")
