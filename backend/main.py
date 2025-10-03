import os
import uuid
import joblib
import time
import pandas as pd
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.exceptions import NotFittedError

# --- App Initialization ---
app = FastAPI(
    title="Exoplanet Classifier API",
    description="An API to train and use an ML model for exoplanet classification.",
    version="1.0.0"
)

# --- CORS Configuration ---
# Allows the frontend (running on a different port) to communicate with this backend.
origins = [
    "http://localhost",
    "http://localhost:5173", # Default Vite dev server port
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory Setup ---
# Create a temporary directory to store session-specific data and models.
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Pydantic Models for API Data Validation ---
class HyperParams(BaseModel):
    n_estimators: int = Field(100, gt=0, description="Number of trees in the forest.")
    max_depth: int | None = Field(None, gt=0, description="Maximum depth of the tree.")

class TrainResponse(BaseModel):
    session_id: str
    metrics: dict
    feature_names: list[str]

class PredictRequest(BaseModel):
    session_id: str
    features: List[float]

class PredictResponse(BaseModel):
    prediction: str
    prediction_label: int

# --- Helper Functions ---
def cleanup_files(session_id: str):
    """Removes files associated with a session to save space."""
    time.sleep(3600) # Delay for 1 hour
    data_path = os.path.join(DATA_DIR, f"{session_id}_data.csv")
    model_path = os.path.join(DATA_DIR, f"{session_id}_model.joblib")
    if os.path.exists(data_path):
        os.remove(data_path)
    if os.path.exists(model_path):
        os.remove(model_path)

# --- API Endpoints ---
@app.post("/upload_and_train", response_model=TrainResponse)
async def upload_and_train(background_tasks: BackgroundTasks, file: UploadFile = File(...), params: HyperParams = Depends()):
    """
    Handles file upload, data preprocessing, model training, and returns metrics.
    This is a single, efficient endpoint for the primary workflow.
    """
    session_id = str(uuid.uuid4())
    data_path = os.path.join(DATA_DIR, f"{session_id}_data.csv")
    model_path = os.path.join(DATA_DIR, f"{session_id}_model.joblib")

    # 1. Save uploaded file
    try:
        contents = await file.read()
        with open(data_path, "wb") as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail="Error saving uploaded file.")
    
    # 2. Load and Preprocess Data
    try:
        df = pd.read_csv(data_path)
        
        # --- Critical Preprocessing Step ---
        # Select relevant features based on domain knowledge of exoplanet data.
        # This makes the model more robust and prevents errors from irrelevant columns.
        features_to_use = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
        target_column = 'koi_disposition'

        if target_column not in df.columns or not all(f in df.columns for f in features_to_use):
            raise ValueError("CSV must contain 'koi_disposition' and all feature columns.")

        df_processed = df[features_to_use + [target_column]].copy()
        
        # Handle missing values simply for the MVP
        df_processed.dropna(inplace=True)

        # Encode the target label (0 for non-exoplanets, 1 for confirmed exoplanets)
        df_processed['label'] = df_processed[target_column].apply(lambda x: 1 if x == 'CONFIRMED' else 0)

        X = df_processed[features_to_use]
        y = df_processed['label']

    except (FileNotFoundError, ValueError) as e:
        cleanup_files(session_id)
        raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")

    # 3. Train the Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        random_state=42,
        n_jobs=-1 # Use all available CPU cores
    )
    model.fit(X_train, y_train)

    # 4. Save the trained model
    joblib.dump(model, model_path)

    # 5. Evaluate and return metrics
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Schedule old files for cleanup after a delay (e.g., 1 hour)
    background_tasks.add_task(cleanup_files, session_id)

    return {
        "session_id": session_id,
        "metrics": report,
        "feature_names": features_to_use
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Makes a prediction using a previously trained model for a given session.
    """
    model_path = os.path.join(DATA_DIR, f"{req.session_id}_model.joblib")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model for this session not found. Please train a model first.")

    try:
        model = joblib.load(model_path)
        
        # The number of features in the request must match the model's expected input
        if len(req.features) != model.n_features_in_:
            raise ValueError(f"Invalid number of features. Expected {model.n_features_in_}, got {len(req.features)}.")

        prediction_label = model.predict([req.features])[0]
        prediction_text = "Confirmed Exoplanet" if prediction_label == 1 else "Not an Exoplanet"

        return {"prediction": prediction_text, "prediction_label": int(prediction_label)}
    
    except (NotFittedError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {str(e)}")