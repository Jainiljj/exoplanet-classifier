import os
import uuid
import joblib
import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional

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
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory Setup ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Pydantic Models for API Data Validation ---
class HyperParams(BaseModel):
    n_estimators: int = Field(100, gt=0, description="Number of trees in the forest.")
    max_depth: Optional[int] = Field(None, gt=0, description="Maximum depth of the tree.")

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
def cleanup_files_now(session_id: str):
    """Removes files associated with a session immediately."""
    data_path = os.path.join(DATA_DIR, f"{session_id}_data.csv")
    model_path = os.path.join(DATA_DIR, f"{session_id}_model.joblib")
    if os.path.exists(data_path):
        try:
            os.remove(data_path)
        except Exception:
            pass
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
        except Exception:
            pass

async def delayed_cleanup_files(session_id: str, delay_seconds: int = 3600):
    """Asynchronously cleans up session files after a specified delay without blocking threads."""
    await asyncio.sleep(delay_seconds)
    cleanup_files_now(session_id)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "online", "message": "Exoplanet Classifier API is running."}

@app.post("/upload_and_train", response_model=TrainResponse)
async def upload_and_train(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    params: HyperParams = Depends()
):
    """
    Handles file upload, data preprocessing, model training, and returns metrics.
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
        
        # Trim column whitespace if present
        df.columns = df.columns.str.strip()

        features_to_use = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
        target_column = 'koi_disposition'

        if target_column not in df.columns or not all(f in df.columns for f in features_to_use):
            missing = [f for f in features_to_use + [target_column] if f not in df.columns]
            raise ValueError(f"CSV missing required column(s): {', '.join(missing)}")

        df_processed = df[features_to_use + [target_column]].copy()
        
        # Handle missing values simply for the MVP
        df_processed.dropna(inplace=True)

        if df_processed.empty:
            raise ValueError("Dataset has no valid rows after removing missing values.")

        # Encode target label (1 for CONFIRMED, 0 otherwise)
        df_processed['label'] = df_processed[target_column].apply(
            lambda x: 1 if str(x).strip().upper() == 'CONFIRMED' else 0
        )

        X = df_processed[features_to_use]
        y = df_processed['label']

    except (FileNotFoundError, ValueError) as e:
        cleanup_files_now(session_id)
        raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")
    except Exception as e:
        cleanup_files_now(session_id)
        raise HTTPException(status_code=500, detail=f"Unexpected error processing data: {str(e)}")

    # 3. Train the Model
    try:
        # Check class distribution for stratification
        stratify_param = y if len(y.unique()) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
        
        model = RandomForestClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # 4. Save the trained model
        joblib.dump(model, model_path)

        # 5. Evaluate and return metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Schedule cleanup after 1 hour (3600 seconds) without blocking response
        background_tasks.add_task(delayed_cleanup_files, session_id, 3600)

        return {
            "session_id": session_id,
            "metrics": report,
            "feature_names": features_to_use
        }
    except Exception as e:
        cleanup_files_now(session_id)
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


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
        
        if len(req.features) != model.n_features_in_:
            raise ValueError(f"Invalid number of features. Expected {model.n_features_in_}, got {len(req.features)}.")

        prediction_label = model.predict([req.features])[0]
        prediction_text = "Confirmed Exoplanet" if int(prediction_label) == 1 else "Not an Exoplanet"

        return {"prediction": prediction_text, "prediction_label": int(prediction_label)}
    
    except (NotFittedError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
