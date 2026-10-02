"""
app.py - Intelligent Disease Prediction System (website version)
==================================================================
A single Python file that does EVERYTHING the command-line version did,
but serves it as a real website using Flask:

  1. DATASET   - builds the synthetic symptom_disease.csv (200 records)
                 if it is not already there (this replaces the old,
                 separate generate_dataset.py file).
  2. MODEL     - trains and compares 4 classifiers, keeps the best one,
                 and re-trains automatically if the CSV ever changes.
  3. DATABASE  - saves every patient and every prediction to SQLite
                 (patients.db) using plain SQL.
  4. WEBSITE   - Flask routes that render templates/index.html
                 (structure) styled by static/style.css (look).

FOLDER LAYOUT REQUIRED (Flask expects this exact structure):
    web/
      app.py                 <- this file
      symptom_disease.csv    <- created automatically on first run
      templates/
        index.html
      static/
        style.css

HOW TO RUN
    pip install flask pandas numpy scikit-learn joblib
    python app.py
Then open the address it prints, normally:  http://127.0.0.1:5000

DISCLAIMER: educational project only - NOT a medical device. It must
never replace advice from a qualified doctor.
"""

import difflib
import hashlib
import io
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import (Flask, Response, redirect, render_template, request,
                   session, url_for)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

# ==========================================================================
# PART 1 - CONFIGURATION
# ==========================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "symptom_disease.csv"
MODEL_PATH = BASE_DIR / "disease_model.joblib"
DB_PATH = BASE_DIR / "patients.db"

TOP_N = 4               # how many possible diseases to show
MIN_SYMPTOMS = 2        # minimum symptoms needed for a prediction
LOW_CONFIDENCE = 40.0   # warn the user below this % confidence
HISTORY_LIMIT = 25      # rows shown on the history table

DISCLAIMER = ("This is a decision-support demo, not a diagnosis. Please "
              "see a qualified doctor or visit a hospital for proper "
              "testing and treatment.")

# Everyday / Nigerian-English words mapped onto the symptom names below
SYNONYMS = {
    "catarrh": "runny_nose", "running_nose": "runny_nose",
    "stuffy_nose": "nasal_congestion", "blocked_nose": "nasal_congestion",
    "body_ache": "body_pain", "body_aches": "body_pain",
    "bodyache": "body_pain", "body_pains": "body_pain",
    "tiredness": "fatigue", "tired": "fatigue", "weak": "weakness",
    "high_temperature": "high_fever", "temperature": "fever",
    "vomit": "vomiting", "throwing_up": "vomiting",
    "loose_stool": "diarrhea", "running_stomach": "diarrhea",
    "diarrhoea": "diarrhea", "stomach_pain": "abdominal_pain",
    "stomach_ache": "abdominal_pain", "belle_pain": "abdominal_pain",
    "waist_pain": "back_pain", "backache": "back_pain",
    "shivering": "chills", "cold_chills": "chills", "sweat": "sweating",
    "breathlessness": "shortness_of_breath",
    "difficulty_breathing": "shortness_of_breath",
    "poor_appetite": "loss_of_appetite", "no_appetite": "loss_of_appetite",
    "loss_of_smell": "loss_of_smell_taste", "loss_of_taste": "loss_of_smell_taste",
    "pain_when_urinating": "painful_urination", "burning_urine": "painful_urination",
    "peeing_often": "frequent_urination", "sore_throats": "sore_throat",
    "phlegm": "cough_with_phlegm", "productive_cough": "cough_with_phlegm",
    "dizzy": "dizziness", "headaches": "headache",
}


# ==========================================================================
# PART 2 - SYNTHETIC DATASET  (was generate_dataset.py)
# ==========================================================================
# How many patient records to create per disease - 200 in total. The
# numbers are rough "how common is it" weights (malaria and colds are seen
# more often than TB or COVID-19), so the model learns realistic priors.
SAMPLES_PER_DISEASE = {
    "Malaria": 32, "Common Cold": 26, "Typhoid Fever": 24,
    "Influenza (Flu)": 21, "Gastroenteritis": 21,
    "Urinary Tract Infection": 18, "Pneumonia": 16, "Allergic Rhinitis": 16,
    "Tuberculosis": 13, "COVID-19": 13,
}
NOISE_PROB = 0.04   # chance of an "unrelated" symptom appearing
SEED = 42

SYMPTOMS = [
    "fever", "high_fever", "chills", "sweating", "headache", "body_pain",
    "joint_pain", "fatigue", "weakness", "cough", "dry_cough",
    "cough_with_phlegm", "sore_throat", "runny_nose", "sneezing",
    "nasal_congestion", "shortness_of_breath", "chest_pain",
    "loss_of_appetite", "nausea", "vomiting", "diarrhea", "abdominal_pain",
    "constipation", "night_sweats", "weight_loss", "blood_in_sputum",
    "painful_urination", "frequent_urination", "loss_of_smell_taste",
    "itchy_eyes", "dizziness", "back_pain",
]

# probability that a symptom is present, given the disease
DISEASE_PROFILES = {
    "Malaria": {
        "fever": .90, "high_fever": .60, "chills": .90, "sweating": .80,
        "headache": .85, "body_pain": .70, "joint_pain": .50, "fatigue": .70,
        "weakness": .65, "nausea": .50, "vomiting": .40,
        "loss_of_appetite": .50, "dizziness": .30, "abdominal_pain": .15,
        "diarrhea": .10, "cough": .15, "sore_throat": .10,
    },
    "Typhoid Fever": {
        "fever": .95, "high_fever": .70, "headache": .70, "fatigue": .70,
        "weakness": .70, "loss_of_appetite": .75, "abdominal_pain": .70,
        "constipation": .50, "diarrhea": .35, "nausea": .50, "vomiting": .25,
        "body_pain": .40, "chills": .30, "dry_cough": .25, "weight_loss": .20,
        "sweating": .30,
    },
    "Influenza (Flu)": {
        "fever": .85, "high_fever": .30, "chills": .60, "headache": .70,
        "body_pain": .85, "joint_pain": .40, "fatigue": .80, "weakness": .60,
        "cough": .80, "dry_cough": .65, "sore_throat": .50, "runny_nose": .45,
        "nasal_congestion": .40, "sneezing": .30, "loss_of_appetite": .40,
        "sweating": .35,
    },
    "Common Cold": {
        "runny_nose": .90, "sneezing": .80, "nasal_congestion": .80,
        "sore_throat": .70, "cough": .60, "cough_with_phlegm": .30,
        "headache": .30, "fatigue": .30, "fever": .15, "body_pain": .20,
        "itchy_eyes": .20,
    },
    "Pneumonia": {
        "fever": .85, "high_fever": .50, "chills": .55, "cough": .95,
        "cough_with_phlegm": .80, "shortness_of_breath": .80,
        "chest_pain": .65, "fatigue": .70, "weakness": .60,
        "loss_of_appetite": .50, "sweating": .40, "headache": .30,
        "nausea": .20, "body_pain": .30, "blood_in_sputum": .10,
    },
    "Tuberculosis": {
        "cough": .95, "cough_with_phlegm": .70, "blood_in_sputum": .50,
        "night_sweats": .80, "weight_loss": .80, "fever": .60, "fatigue": .80,
        "weakness": .60, "chest_pain": .50, "loss_of_appetite": .70,
        "shortness_of_breath": .35, "chills": .30,
    },
    "Gastroenteritis": {
        "diarrhea": .95, "vomiting": .75, "nausea": .80, "abdominal_pain": .85,
        "fever": .40, "loss_of_appetite": .60, "weakness": .60, "fatigue": .50,
        "dizziness": .35, "headache": .30, "body_pain": .25, "sweating": .20,
    },
    "Urinary Tract Infection": {
        "painful_urination": .95, "frequent_urination": .90,
        "abdominal_pain": .50, "back_pain": .45, "fever": .35, "chills": .20,
        "nausea": .25, "fatigue": .30, "weakness": .20,
        "loss_of_appetite": .20, "vomiting": .10,
    },
    "COVID-19": {
        "fever": .70, "dry_cough": .75, "cough": .80, "fatigue": .75,
        "body_pain": .60, "headache": .60, "sore_throat": .50,
        "loss_of_smell_taste": .65, "shortness_of_breath": .40, "chills": .30,
        "diarrhea": .20, "nasal_congestion": .30, "runny_nose": .25,
        "loss_of_appetite": .35, "chest_pain": .20,
    },
    "Allergic Rhinitis": {
        "sneezing": .95, "runny_nose": .85, "nasal_congestion": .75,
        "itchy_eyes": .85, "headache": .25, "cough": .30, "sore_throat": .20,
        "fatigue": .20,
    },
}


def _apply_dataset_rules(row: np.ndarray) -> np.ndarray:
    """Keep generated rows logically consistent (high fever implies fever)."""
    idx = {name: i for i, name in enumerate(SYMPTOMS)}
    if row[idx["high_fever"]]:
        row[idx["fever"]] = 1
    if row[idx["dry_cough"]] or row[idx["cough_with_phlegm"]]:
        row[idx["cough"]] = 1
    return row


def build_dataset(samples_per_disease: dict = SAMPLES_PER_DISEASE,
                  seed: int = SEED) -> pd.DataFrame:
    """Randomly generate patient rows from the disease profiles above."""
    rng = np.random.default_rng(seed)
    rows = []
    for disease, profile in DISEASE_PROFILES.items():
        probs = np.array([profile.get(s, NOISE_PROB) for s in SYMPTOMS])
        made, target = 0, samples_per_disease[disease]
        while made < target:
            row = (rng.random(len(SYMPTOMS)) < probs).astype(int)
            row = _apply_dataset_rules(row)
            if row.sum() < 2:            # every patient needs >= 2 symptoms
                continue
            rows.append(list(row) + [disease])
            made += 1
    df = pd.DataFrame(rows, columns=SYMPTOMS + ["disease"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def ensure_dataset() -> None:
    if not DATA_PATH.exists():
        build_dataset().to_csv(DATA_PATH, index=False)


def dataset_fingerprint() -> str:
    """A short 'ID' of the CSV's contents, used to detect when it changes."""
    ensure_dataset()
    return hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()


# ==========================================================================
# PART 3 - PREPROCESSING AND MODEL TRAINING  (was disease_prediction_system.py)
# ==========================================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw CSV so it is safe to train on (Pandas)."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=["disease"])
    df["disease"] = df["disease"].astype(str).str.strip()

    features = [c for c in df.columns if c != "disease"]
    df[features] = df[features].fillna(0).astype(int).clip(0, 1)
    df = df[df[features].sum(axis=1) > 0]
    return df.reset_index(drop=True)


def load_dataset() -> pd.DataFrame:
    ensure_dataset()
    return preprocess(pd.read_csv(DATA_PATH))


def candidate_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Naive Bayes": BernoulliNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }


def train_and_evaluate() -> dict:
    """Compare 4 models with cross-validation, keep the best, test, save it."""
    df = load_dataset()
    features = [c for c in df.columns if c != "disease"]
    X, y = df[features], df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for name, model in candidate_models().items():
        cv_scores[name] = cross_val_score(model, X_train, y_train,
                                          cv=cv, scoring="accuracy").mean()

    best_name = max(cv_scores, key=cv_scores.get)
    model = candidate_models()[best_name]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    # 200 records means a small ~40-row test set, so its accuracy can jump
    # around. A 5-fold check over ALL the records is a steadier number.
    full_scores = cross_val_score(candidate_models()[best_name], X, y,
                                  cv=cv, scoring="accuracy")

    model.fit(X, y)                     # final fit on ALL the data
    bundle = {
        "model": model,
        "features": features,
        "model_name": best_name,
        "accuracy": accuracy,
        "n_test": len(y_test),
        "cv_scores": cv_scores,
        "cv_full_mean": float(full_scores.mean()),
        "cv_full_std": float(full_scores.std()),
        "report": report,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_records": len(df),
        "data_hash": dataset_fingerprint(),
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_or_train_model() -> dict:
    """Reuse the saved model if the CSV hasn't changed, else retrain."""
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if bundle.get("data_hash") == dataset_fingerprint():
                return bundle
        except Exception:
            pass
    return train_and_evaluate()


# ==========================================================================
# PART 4 - SYMPTOM MATCHING AND PREDICTION
# ==========================================================================
def normalize(text: str) -> str:
    return "_".join(text.lower().replace("-", " ").replace("/", " ").split())


def apply_implications(symptoms: list, features: list) -> list:
    result = list(symptoms)
    if "high_fever" in result and "fever" in features and "fever" not in result:
        result.append("fever")
    if ({"dry_cough", "cough_with_phlegm"} & set(result)
            and "cough" in features and "cough" not in result):
        result.append("cough")
    return result


def match_symptoms(raw: str, features: list):
    """Turn free text like 'fever, catarrh, hedache' into known symptoms."""
    matched, unmatched = [], []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key = normalize(item)
        key = SYNONYMS.get(key, key)
        if key in features:
            candidate = key
        else:
            close = difflib.get_close_matches(
                normalize(item), features + list(SYNONYMS), n=1, cutoff=0.8)
            candidate = SYNONYMS.get(close[0], close[0]) if close else None
            if candidate not in features:
                candidate = None
        if candidate is None:
            unmatched.append(item)
        elif candidate not in matched:
            matched.append(candidate)
    return apply_implications(matched, features), unmatched


def predict_disease(bundle: dict, symptoms: list, top_n: int = TOP_N) -> list:
    """Return [(disease, confidence_percent), ...] sorted, best first."""
    model, features = bundle["model"], bundle["features"]
    row = pd.DataFrame([[1 if f in symptoms else 0 for f in features]],
                       columns=features)
    probs = model.predict_proba(row)[0]
    results = []
    for rank, i in enumerate(np.argsort(probs)[::-1][:top_n]):
        if rank == 0 or probs[i] >= 0.005:
            results.append((str(model.classes_[i]), float(probs[i]) * 100))
    return results


def pretty(symptom: str) -> str:
    return symptom.replace("_", " ")


# ==========================================================================
# PART 5 - DATABASE  (SQL with sqlite3)
# ==========================================================================
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS patients (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                age         INTEGER,
                gender      TEXT,
                phone       TEXT,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   INTEGER NOT NULL,
                symptoms     TEXT NOT NULL,
                top_disease  TEXT NOT NULL,
                confidence   REAL NOT NULL,
                all_results  TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            );
        """)


def get_or_create_patient(name, age, gender, phone) -> int:
    with get_db() as conn:
        row = conn.execute(
            """SELECT id FROM patients
               WHERE LOWER(name) = LOWER(?)
                 AND IFNULL(age, -1) = IFNULL(?, -1)
                 AND IFNULL(phone, '') = ?""",
            (name, age, phone or "")).fetchone()
        if row:
            return row[0]
        cur = conn.execute(
            "INSERT INTO patients (name, age, gender, phone, created_at) "
            "VALUES (?, ?, ?, ?, ?)", (name, age, gender, phone or "", now()))
        return cur.lastrowid


def save_prediction(patient_id: int, symptoms: list, results: list) -> None:
    import json
    with get_db() as conn:
        conn.execute(
            """INSERT INTO predictions
               (patient_id, symptoms, top_disease, confidence, all_results, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (patient_id, ", ".join(pretty(s) for s in symptoms),
             results[0][0], results[0][1], json.dumps(results), now()))


def fetch_history(name: str = "", limit: int = None) -> pd.DataFrame:
    query = """
        SELECT p.id AS record_id, p.created_at AS date, pt.name AS patient_name,
               pt.age, pt.gender, p.symptoms,
               p.top_disease, ROUND(p.confidence, 1) AS confidence_pct
        FROM predictions p
        JOIN patients pt ON pt.id = p.patient_id
        WHERE LOWER(pt.name) LIKE ?
        ORDER BY p.id DESC
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    with get_db() as conn:
        return pd.read_sql_query(query, conn, params=(f"%{name.lower()}%",))


def disease_distribution() -> list:
    """[(disease, count, pct_of_max), ...] from the training data - for the
    'Cases in the training data' chart, drawn with plain CSS bars."""
    counts = load_dataset()["disease"].value_counts()
    top = counts.max()
    return [(d, int(c), round(c / top * 100, 1)) for d, c in counts.items()]


def history_disease_summary() -> list:
    """Same shape, but built from saved predictions instead of training data."""
    df = fetch_history()
    if df.empty:
        return []
    counts = df["top_disease"].value_counts()
    top = counts.max()
    return [(d, int(c), round(c / top * 100, 1)) for d, c in counts.items()]


# ==========================================================================
# PART 6 - FLASK WEBSITE
# ==========================================================================
app = Flask(__name__)
app.secret_key = "siwes-disease-prediction-demo"  # fine for a student demo;
                                                   # use a random secret in production

init_db()
BUNDLE = load_or_train_model()
FEATURES = BUNDLE["features"]


@app.route("/", methods=["GET"])
def index():
    q = request.args.get("q", "").strip()
    history_df = fetch_history(q, limit=HISTORY_LIMIT)
    history_df["age"] = history_df["age"].apply(
        lambda a: int(a) if pd.notna(a) else None)
    history_records = history_df.to_dict("records")

    result = session.pop("last_result", None)
    error = session.pop("error", None)
    notice = session.pop("notice", None)

    return render_template(
        "index.html",
        features=FEATURES,
        pretty=pretty,
        symptom_count=len(FEATURES),
        min_symptoms=MIN_SYMPTOMS,
        low_confidence=LOW_CONFIDENCE,
        disclaimer=DISCLAIMER,
        result=result,
        error=error,
        notice=notice,
        q=q,
        history=history_records,
        history_count=len(history_records),
        distribution=disease_distribution(),
        history_summary=history_disease_summary(),
        model_name=BUNDLE["model_name"],
        accuracy=round(BUNDLE["accuracy"] * 100, 1),
        cv_mean=round(BUNDLE["cv_full_mean"] * 100, 1),
        cv_std=round(BUNDLE["cv_full_std"] * 100, 1),
        n_records=BUNDLE["n_records"],
        trained_at=BUNDLE["trained_at"],
        cv_scores=[(n, round(s * 100, 1)) for n, s in BUNDLE["cv_scores"].items()],
    )


@app.route("/predict", methods=["POST"])
def predict():
    name = request.form.get("name", "").strip()
    age_raw = request.form.get("age", "").strip()
    gender = request.form.get("gender", "").strip()
    phone = request.form.get("phone", "").strip()
    chosen = request.form.getlist("symptom")
    typed = request.form.get("quick_symptoms", "").strip()

    if not name:
        session["error"] = "Please enter the patient's name."
        return redirect(url_for("index"))

    age = None
    if age_raw:
        if age_raw.isdigit() and 0 <= int(age_raw) <= 120:
            age = int(age_raw)
        else:
            session["error"] = "Age must be a number from 0 to 120."
            return redirect(url_for("index"))

    matched, unknown = match_symptoms(typed, FEATURES) if typed else ([], [])
    symptoms = apply_implications(sorted(set(chosen) | set(matched)), FEATURES)

    if len(symptoms) < MIN_SYMPTOMS:
        session["error"] = (f"Please select or type at least {MIN_SYMPTOMS} "
                            "recognised symptoms.")
        return redirect(url_for("index"))

    results = predict_disease(BUNDLE, symptoms)
    patient_id = get_or_create_patient(name, age, gender, phone)
    save_prediction(patient_id, symptoms, results)

    session["last_result"] = {
        "patient": name,
        "symptoms": [pretty(s) for s in symptoms],
        "results": results,
        "unknown": unknown,
        "low_confidence": results[0][1] < LOW_CONFIDENCE,
    }
    return redirect(url_for("index"))


@app.route("/retrain", methods=["POST"])
def retrain():
    global BUNDLE, FEATURES
    BUNDLE = train_and_evaluate()
    FEATURES = BUNDLE["features"]
    session["notice"] = (f"Model retrained: {BUNDLE['model_name']} "
                         f"({BUNDLE['accuracy'] * 100:.1f}% hold-out accuracy).")
    return redirect(url_for("index"))


@app.route("/export.csv")
def export_csv():
    q = request.args.get("q", "").strip()
    df = fetch_history(q)
    if df.empty:
        session["error"] = "There are no records to export yet."
        return redirect(url_for("index"))
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    filename = f"prediction_report_{datetime.now():%Y%m%d_%H%M%S}.csv"
    return Response(
        buffer.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"})


if __name__ == "__main__":
    print(f"\nModel ready: {BUNDLE['model_name']} "
          f"({BUNDLE['accuracy'] * 100:.1f}% hold-out accuracy, "
          f"{BUNDLE['n_records']} training records)")
    print(f"{DISCLAIMER}\n")
    app.run(debug=True)