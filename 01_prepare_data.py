"""
Step 1: Clean the structured CSV and convert every patient row into a
short clinical note (text). The notes are the input for the NLP models.
Output: data/clinical_notes.csv  (columns: text, label)
"""
import pandas as pd

# keep_default_na=False  -> the word "None" (no symptom) is NOT treated as missing
df = pd.read_csv("data/pneumonia_dataset.csv", keep_default_na=False)
df = df.replace("-", "")  # "-" means value not recorded

# Xray and WBC_count are ~99% empty, so they are dropped
df = df.drop(columns=["Xray", "WBC_count", "PatientID"])

df["Oxygen_saturation"] = pd.to_numeric(df["Oxygen_saturation"], errors="coerce")


def symptom(name, value, none_text):
    """Turn a symptom level into a phrase. Handles negation ('no ...')."""
    v = str(value).strip()
    if v == "" or v.lower() == "none":
        return none_text
    return f"{v.lower()} {name}"


def make_note(r):
    sex = "male" if r["Gender"] == "M" else "female"
    parts = [f"{r['Age']} year old {sex} patient."]

    # cough
    c = str(r["Cough"]).strip()
    parts.append("no cough." if c in ("", "None") else f"presents with {c.lower()} cough.")
    # fever
    f = str(r["Fever"]).strip()
    parts.append("no fever." if f in ("", "None") else f"{f.lower()} fever reported.")
    # shortness of breath
    s = str(r["Shortness_of_breath"]).strip()
    parts.append("denies shortness of breath." if s in ("", "None")
                 else f"{s.lower()} shortness of breath.")
    # chest pain
    p = str(r["Chest_pain"]).strip()
    parts.append("no chest pain." if p in ("", "None") else f"{p.lower()} chest pain.")
    # fatigue
    t = str(r["Fatigue"]).strip()
    parts.append("no fatigue." if t in ("", "None") else f"{t.lower()} fatigue.")
    # confusion
    if r["Confusion"] == "Yes":
        parts.append("patient is confused.")
    elif r["Confusion"] == "No":
        parts.append("patient is not confused.")
    # oxygen saturation
    o = r["Oxygen_saturation"]
    if pd.notna(o):
        level = "low" if o < 92 else "normal"
        parts.append(f"oxygen saturation {int(o)} percent, {level}.")
    # crackles
    parts.append("crackles heard on auscultation." if r["Crackles"] == "Yes"
                 else "no crackles on auscultation.")
    # sputum
    sp = str(r["Sputum_color"]).strip()
    if sp != "":
        parts.append(f"{sp.lower()} sputum.")
    # temperature
    temp = r["Temperature"]
    parts.append(f"temperature {temp} degrees celsius.")
    return " ".join(parts)


df["text"] = df.apply(make_note, axis=1)
df["label"] = (df["Diagnosis"] == "Yes").astype(int)

df[["text", "label"]].to_csv("data/clinical_notes.csv", index=False)
print("Saved data/clinical_notes.csv", df.shape)
print("Class balance:\n", df["label"].value_counts())
print("\nExample note:\n", df["text"].iloc[0])
