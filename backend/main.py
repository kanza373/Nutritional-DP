"""
main.py - FastAPI backend for the Nutritional Deficiency Predictor.

Loads the RandomForest models + StandardScaler + LabelEncoders produced by
train.py, exposes POST /api/predict, and serves the static frontend.

AUTO-RETRAIN: a background thread watches train.py for changes. The moment
you save an edit to train.py, it automatically re-runs the training
pipeline and hot-swaps the new models into the running server - no manual
`python train.py` or server restart needed. Progress is exposed at
GET /api/model-status, and the frontend polls it to show a live badge.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in a browser.
"""

import io
import os
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# --- OCR feature: optional imports -----------------------------------------
# Wrapped in try/except so the rest of the app (existing ML model, existing
# endpoints, existing UI) keeps working even if pytesseract / the tesseract
# binary isn't installed yet - the /api/ocr-extract endpoint will just return
# a clear error telling the user to install it, instead of the server crashing.
# try:
#     import pytesseract
#     from PIL import Image
#     OCR_AVAILABLE = True
# except ImportError:
#     OCR_AVAILABLE = False
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    from PIL import Image

    # Windows par tesseract.exe ka path batana parta hai.
    # Agar aapne kisi aur jagah install kiya hai to ye line badal dein.
    import platform
    if platform.system() == "Windows":
        _TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(_TESSERACT_EXE).exists():
            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_EXE

    OCR_AVAILABLE = True
except ImportError as _exc:
    print(f"[startup] OCR libraries load nahi huin: {_exc}")
    OCR_AVAILABLE = False

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR.parent / "frontend"
TRAIN_SCRIPT = BASE_DIR / "train.py"
NOTEBOOK_PATH = BASE_DIR / "ICD_Project.ipynb"
WATCHED_FILE = NOTEBOOK_PATH  # editing THIS file triggers an automatic retrain

REQUIRED_MODEL_FILES = [
    "Iron_Deficiency_Risk_model.joblib",
    "Calcium_Deficiency_Risk_model.joblib",
    "Vitamin_D_Deficiency_Risk_model.joblib",
    "Vitamin_B12_Deficiency_Risk_model.joblib",
    "scaler.joblib",
    "label_encoders.joblib",
    "feature_order.json",
]

RISK_KEY_MAP = {
    "Iron_Deficiency_Risk": "iron",
    "Calcium_Deficiency_Risk": "calcium",
    "Vitamin_D_Deficiency_Risk": "vitamin_d",
    "Vitamin_B12_Deficiency_Risk": "vitamin_b12",
}

# ---------------------------------------------------------------------------
# Live artifact store. Everything the API needs lives in this one dict so a
# retrain can atomically swap it out while requests are being served.
# ---------------------------------------------------------------------------
ARTIFACTS = {}
ARTIFACTS_LOCK = threading.RLock()

MODEL_STATE = {
    "status": "starting",       # starting | idle | training | error
    "last_trained_at": None,    # ISO timestamp of the last successful train
    "last_error": None,
    "version": 0,                # increments on every successful (re)load
}
STATE_LOCK = threading.RLock()


def _set_state(**kwargs):
    with STATE_LOCK:
        MODEL_STATE.update(kwargs)


def load_artifacts():
    """(Re)load every model artifact from disk into ARTIFACTS."""
    missing = [f for f in REQUIRED_MODEL_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        raise RuntimeError(f"Missing model artifacts {missing} in {MODELS_DIR}.")

    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    label_encoders = joblib.load(MODELS_DIR / "label_encoders.joblib")
    feature_order = json.loads((MODELS_DIR / "feature_order.json").read_text())
    metrics_path = MODELS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    models = {
        target: joblib.load(MODELS_DIR / f"{target}_model.joblib")
        for target in RISK_KEY_MAP
    }

    with ARTIFACTS_LOCK:
        ARTIFACTS["scaler"] = scaler
        ARTIFACTS["label_encoders"] = label_encoders
        ARTIFACTS["feature_order"] = feature_order
        ARTIFACTS["metrics"] = metrics
        ARTIFACTS["models"] = models

    with STATE_LOCK:
        MODEL_STATE["version"] += 1


# Initial load at import time so the app fails fast if artifacts are missing
# (or trains them once automatically if train.py exists but models don't).
try:
    load_artifacts()
    _set_state(status="idle", last_trained_at=datetime.now(timezone.utc).isoformat())
except RuntimeError:
    if TRAIN_SCRIPT.exists() and NOTEBOOK_PATH.exists():
        print("[startup] No model artifacts found - running the notebook once ...")
        subprocess.run([sys.executable, str(TRAIN_SCRIPT)], cwd=str(BASE_DIR), check=True)
        load_artifacts()
        _set_state(status="idle", last_trained_at=datetime.now(timezone.utc).isoformat())
    else:
        raise


# ---------------------------------------------------------------------------
# Background watcher: polls train.py's mtime, retrains + hot-reloads on change
# ---------------------------------------------------------------------------
def watch_train_script(poll_seconds: float = 1.5, debounce_seconds: float = 1.2):
    if not WATCHED_FILE.exists():
        print(f"[watcher] {WATCHED_FILE} not found - auto-retrain disabled.")
        return
    last_mtime = WATCHED_FILE.stat().st_mtime

    while True:
        time.sleep(poll_seconds)
        try:
            mtime = WATCHED_FILE.stat().st_mtime
        except FileNotFoundError:
            continue

        if mtime == last_mtime:
            continue

        # Debounce: wait for the editor / Jupyter to finish writing the file.
        time.sleep(debounce_seconds)
        try:
            mtime = WATCHED_FILE.stat().st_mtime
        except FileNotFoundError:
            continue
        last_mtime = mtime

        print(f"[watcher] {WATCHED_FILE.name} changed - re-executing notebook & retraining ...")
        _set_state(status="training", last_error=None)
        try:
            result = subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT)],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-2000:] or "train.py exited with an error")

            load_artifacts()
            _set_state(
                status="idle",
                last_trained_at=datetime.now(timezone.utc).isoformat(),
                last_error=None,
            )
            print("[watcher] retrain complete, new model is live.")
        except Exception as exc:
            _set_state(status="error", last_error=str(exc))
            print(f"[watcher] retrain FAILED, keeping previous model live: {exc}")


class PatientInput(BaseModel):
    Age: int = Field(..., ge=1, le=120)
    Gender: Literal["Male", "Female"]
    Diet_Type: Literal["Vegetarian", "Non-Vegetarian", "Vegan"]
    Calories: float = Field(..., ge=0)
    Protein: float = Field(..., ge=0)
    Vegetable_Intake: float = Field(..., ge=0)
    Fruit_Intake: float = Field(..., ge=0)
    Dairy_Intake: float = Field(..., ge=0)
    Water_Intake: float = Field(..., ge=0)
    Meal_Frequency: int = Field(..., ge=1, le=15)
    BMI: float = Field(..., ge=5, le=80)
    Physical_Activity: Literal["Low", "Moderate", "High"]


RISK_CLASS_LABELS = {0: "Low", 1: "Moderate", 2: "High"}


def run_inference(payload: PatientInput):
    row = payload.dict()
    df = pd.DataFrame([row])

    # The notebook collapses Diet_Type into just "Veg"/"Non-Veg" (losing the
    # Vegan/Vegetarian distinction), and separately derives Is_Vegan /
    # Is_Vegetarian flags from the ORIGINAL 3-way Diet_Type before that
    # collapse happens. The API still accepts the original 3-way value from
    # the frontend, so we have to reproduce both derivations here before
    # handing the row to the encoders/model - otherwise Diet_Type='Vegan' or
    # 'Vegetarian' is an unseen category for the encoder, and Is_Vegan /
    # Is_Vegetarian are simply missing columns.
    original_diet_type = df.at[0, "Diet_Type"]
    df["Is_Vegan"] = 1 if original_diet_type == "Vegan" else 0
    df["Is_Vegetarian"] = 1 if original_diet_type == "Vegetarian" else 0
    df["Diet_Type"] = "Veg" if original_diet_type in ("Vegan", "Vegetarian") else "Non-Veg"

    with ARTIFACTS_LOCK:
        label_encoders = ARTIFACTS["label_encoders"]
        feature_order = ARTIFACTS["feature_order"]
        scaler = ARTIFACTS["scaler"]
        models = ARTIFACTS["models"]

    for col, le in label_encoders.items():
        value = df.at[0, col]
        if value not in le.classes_:
            raise HTTPException(
                status_code=422,
                detail=f"Unseen category '{value}' for feature '{col}'. Expected one of {list(le.classes_)}.",
            )
        df[col] = le.transform(df[col])

    missing_features = [f for f in feature_order if f not in df.columns]
    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=f"Server-side feature mismatch, missing: {missing_features}. Retrain the model.",
        )
    df = df[feature_order]
    X_scaled = scaler.transform(df)

    results = {}
    for target, model in models.items():
        proba = model.predict_proba(X_scaled)[0]
        pred_class = int(model.predict(X_scaled)[0])
        # proba[pred_class] is the model's confidence in whichever class
        # it actually predicted - this maps directly to the notebook's own
        # 0=Low / 1=Moderate / 2=High target encoding, unlike the previous
        # version which always read proba[1] regardless of pred_class and
        # could show "Low" risk even when the model had predicted class 2.
        confidence = float(proba[pred_class]) if pred_class < len(proba) else float(max(proba))
        results[RISK_KEY_MAP[target]] = {
            "target": target,
            "risk_level": RISK_CLASS_LABELS.get(pred_class, "Moderate"),
            "probability": round(confidence, 4),
            "predicted_class": pred_class,
        }
    return results

# ---------------------------------------------------------------------------
# OCR feature - "Upload Image to Auto-Fill Fields"
#
# Ye section poori tarah additive hai: ARTIFACTS, trained models,
# run_inference(), ya /api/predict ka contract kisi tarah nahi badalta.
# Sirf uploaded image se text parhta hai aur NLP pipeline se form ke
# fields guess karta hai.
#
# Extraction ka asal logic nlp_extraction.py mein hai, jo notebook ke
# "Section 16: NLP-Based OCR Text Extraction" mein develop hua hai -
# backend aur notebook dono ek hi source of truth use karte hain.
# ---------------------------------------------------------------------------
try:
    from nlp_extraction import (
        extract_fields_from_text,
        ALL_FIELDS as OCR_ALL_FIELDS,
        SPACY_AVAILABLE,
    )
    NLP_AVAILABLE = True
except ImportError as _exc:
    print(f"[startup] nlp_extraction.py load nahi hua: {_exc}")
    NLP_AVAILABLE = False
    SPACY_AVAILABLE = False
    OCR_ALL_FIELDS = []


app = FastAPI(title="Nutritional Deficiency Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def start_watcher():
    # Vercel functions are short-lived/serverless; do not start a permanent watcher.
    # Local development can still use the original auto-retrain watcher.
    if os.environ.get("VERCEL"):
        return
    thread = threading.Thread(target=watch_train_script, daemon=True)
    thread.start()


@app.post("/api/predict")
def predict(payload: PatientInput):
    try:
        return {"ok": True, "results": run_inference(payload)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/api/ocr-extract")
async def ocr_extract(image: UploadFile = File(...)):
    """
    OCR feature endpoint - reads an uploaded image (diet chart, nutrition
    label, health report, etc.), extracts text, and tries to map values onto
    the EXISTING form fields. This never touches ARTIFACTS / the trained
    models and never calls run_inference() - it only returns data for the
    frontend to pre-fill into the form, which the user must still review and
    submit themselves via the existing "Predict Risk" button.
    """
    if not OCR_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail=(
                "OCR is not available on this server. Install the OCR dependencies "
                "(pip install pytesseract Pillow) and the Tesseract OCR engine "
                "(e.g. `apt install tesseract-ocr` on Linux, or download it for "
                "Windows/Mac), then restart the server."
            ),
        )

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (PNG, JPG, etc.).")

    try:
        raw_bytes = await image.read()
        img = Image.open(io.BytesIO(raw_bytes))
        img = img.convert("L")  # grayscale - improves OCR accuracy on most labels/reports
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image file. Please try a different image.")

    try:
        text = pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=(
                "The Tesseract OCR engine isn't installed on this server. "
                "Install it (e.g. `apt install tesseract-ocr`) and restart the server."
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed to process the image: {exc}")

    if not text or not text.strip():
        return {
            "ok": True,
            "extracted": {},
            "needs_confirmation": {},
            "details": {},
            "detected_fields": [],
            "not_detected_fields": OCR_ALL_FIELDS,
            "engine": "spacy+rules" if SPACY_AVAILABLE else "rules",
            "message": "Could not read any text from this image. Please enter the fields manually.",
        }

    if not NLP_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="NLP extraction module (nlp_extraction.py) load nahi hua. Check server logs.",
        )

    # NLP pipeline - normalization -> tokenization -> lexicon matching ->
    # unit conversion -> confidence scoring. Koi ML model involved nahi.
    result = extract_fields_from_text(text)

    auto = result["auto"]                 # confidence zyada -> field bhar do
    confirm = result["confirm"]           # shak hai -> user confirm kare
    details = result["details"]           # confidence / unit / evidence
    not_detected_fields = result["not_detected"]

    detected_fields = list(auto.keys())

    if detected_fields and not not_detected_fields and not confirm:
        message = "Information extracted successfully. Please review the fields before prediction."
    elif detected_fields or confirm:
        message = "Some fields could not be detected. Please enter them manually."
    else:
        message = "No matching information could be detected in this image. Please enter the fields manually."

    return {
        "ok": True,
        "extracted": auto,
        "needs_confirmation": confirm,
        "details": details,
        "detected_fields": detected_fields,
        "not_detected_fields": not_detected_fields,
        "engine": "spacy+rules" if SPACY_AVAILABLE else "rules",
        "message": message,
    }


@app.get("/api/metrics")
def get_metrics():
    with ARTIFACTS_LOCK:
        return ARTIFACTS["metrics"]


@app.get("/api/features")
def get_features():
    with ARTIFACTS_LOCK:
        return {
            "feature_order": ARTIFACTS["feature_order"],
            "categorical_options": {
                col: list(le.classes_) for col, le in ARTIFACTS["label_encoders"].items()
            },
        }


@app.get("/api/health")
def health():
    with ARTIFACTS_LOCK:
        loaded = list(ARTIFACTS["models"].keys())
    return {"status": "ok", "models_loaded": loaded}


@app.get("/api/model-status")
def model_status():
    with STATE_LOCK:
        return dict(MODEL_STATE)


# Serve the frontend (index.html + assets) from the same origin so there is
# no CORS/config step for the user - one `uvicorn` command runs everything.
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")
