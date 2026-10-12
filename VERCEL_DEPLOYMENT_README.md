# Vercel deployment notes

This package has been rearranged for Vercel: `api/index.py` exposes the FastAPI app, `vercel.json` routes requests to it, and `requirements.txt` is at the repository root.

## Deploy
1. Extract this ZIP.
2. Upload/push the **contents** of this folder to a GitHub repository (do not upload the ZIP as a file).
3. In Vercel, import that repository. Set Root Directory to the repository root (the folder containing `vercel.json`, `api/`, `backend/`, and `requirements.txt`).
4. Leave Framework Preset as Other; do not set a separate build command or output directory. Deploy.
5. After deployment, open `/api/health` to verify the model API is up.

## Important limitations
- Vercel is serverless; model retraining/background file watching is disabled there. Retrain models locally and redeploy updated model files.
- The OCR endpoint requires the native Tesseract OCR executable, which is not installed by Python `pytesseract` alone. OCR may return an error on Vercel until a compatible deployment strategy for the Tesseract binary is configured.
- Machine-learning dependencies and model files make this a relatively heavy deployment. If Vercel rejects the function bundle, exceeds function limits, or has cold-start/memory issues, host the FastAPI backend on a Python-friendly service and deploy the static frontend separately.
