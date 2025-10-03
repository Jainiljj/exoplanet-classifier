🚀 Exoplanet Classifier - AI/ML MVP Blueprint
This document contains the complete specification for building a web-based, interactive AI/ML tool for exoplanet classification, as per the NASA Space Apps Challenge. It is designed to be used by the Gemini CLI to generate the entire project in a single operation.

🎯 Core Objective
Create a web application where a user can upload a dataset of potential exoplanet candidates, train a machine learning model with tunable hyperparameters, evaluate its performance, and use the trained model to predict whether new, user-provided data points are exoplanets.

✅ MVP Feature Set
Stateless Session Management: Each user session is unique and isolated, preventing data and model conflicts between concurrent users.

Dataset Upload: Users can upload a CSV file containing exoplanet data.

Interactive Model Training: Users can adjust key hyperparameters (like n_estimators and max_depth) and trigger a training process on the backend.

Performance Metrics Display: After training, the frontend displays key classification metrics (accuracy, precision, recall, F1-score) returned by the backend.

Dynamic Prediction Interface: The UI dynamically generates input fields based on the features the model was trained on, allowing users to input new data for prediction in a structured way.

Real-time Feedback: The UI provides loading indicators and clear error messages for a smooth user experience.

🛠️ Tech Stack
Backend: FastAPI (Python)

Machine Learning: Scikit-learn, Pandas

Frontend: React (Vite)

API Communication: REST API (JSON)

📂 Final Project Structure
The complete project will be organized into a backend and a frontend directory.

/exoplanet-classifier-project
├── /backend
│   ├── main.py             # FastAPI application logic
│   ├── requirements.txt      # Python dependencies
│   └── .gitignore            # Git ignore for Python
│
└── /frontend
    ├── src
    │   ├── App.jsx           # Main React component
    │   ├── App.css           # Styling for the component
    │   └── main.jsx          # React app entry point
    ├── index.html            # HTML root
    ├── package.json          # Node dependencies and scripts
    └── .gitignore            # Git ignore for Node

backend req:
fastapi
uvicorn[standard]
scikit-learn
pandas
joblib
python-multipart

backend/.gitignore:
pycache/
*.pyc
.env
/data/

This is the core of the backend. It handles stateless session management using UUIDs, file uploads, data preprocessing, model training, and prediction via robust, error-handled FastAPI endpoints.

backend/main.py

import os
import uuid
import joblib
import pandas as pd
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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
    "[http://127.0.0.1:5173](http://127.0.0.1:5173)",
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
    data_path = os.path.join(DATA_DIR, f"{session_id}_data.csv")
    model_path = os.path.join(DATA_DIR, f"{session_id}_model.joblib")
    if os.path.exists(data_path):
        os.remove(data_path)
    if os.path.exists(model_path):
        os.remove(model_path)

# --- API Endpoints ---
@app.post("/upload_and_train", response_model=TrainResponse)
async def upload_and_train(background_tasks: BackgroundTasks, params: HyperParams, file: UploadFile = File(...)):
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

frontend/package.json:

{
  "name": "exoplanet-classifier-ui",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.15",
    "@types/react-dom": "^18.2.7",
    "@vitejs/plugin-react": "^4.0.3",
    "vite": "^4.4.5"
  }
}

frontend/.gitignore:

Logs
logs
.log
npm-debug.log
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*

Runtime data
pids
*.pid
*.seed
*.pid.lock

Directory for instrumented libs generated by jscoverage/JSCover
lib-cov

Coverage directory used by tools like istanbul
coverage
*.lcov

nyc test coverage
.nyc_output

Grunt intermediate storage (http://gruntjs.com/creating-plugins#storing-task-files)
.grunt

Bower dependency directory (https://bower.io/)
bower_components

node-waf configuration
.lock-wscript

Compiled binary addons (https://nodejs.org/api/addons.html)
build/Release

Dependency directories
node_modules/
dist
dist-ssr

Vite env variables
.env*.local

VSCode settings
.vscode

frontend/index.html:
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Exoplanet Classifier</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>

Self-Contained React App:

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

frontend/src/App.css:

:root {
  --primary-color: #0d6efd;
  --secondary-color: #6c757d;
  --bg-color: #f8f9fa;
  --card-bg: #ffffff;
  --font-color: #212529;
  --border-color: #dee2e6;
  --success-color: #198754;
  --danger-color: #dc3545;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  margin: 0;
  background-color: var(--bg-color);
  color: var(--font-color);
  display: flex;
  justify-content: center;
  align-items: flex-start;
  min-height: 100vh;
  padding: 2rem;
}

#root {
  width: 100%;
  max-width: 900px;
}

.App {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.header {
  text-align: center;
  border-bottom: 2px solid var(--primary-color);
  padding-bottom: 1rem;
}

.header h1 {
  margin: 0;
  font-size: 2.5rem;
  color: var(--primary-color);
}

.header p {
  margin-top: 0.5rem;
  font-size: 1.1rem;
  color: var(--secondary-color);
}

.card {
  background-color: var(--card-bg);
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  padding: 1.5rem;
  border: 1px solid var(--border-color);
}

.card h2 {
  margin-top: 0;
  color: var(--primary-color);
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 0.5rem;
  margin-bottom: 1rem;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

input[type="file"], input[type="number"], input[type="text"] {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  box-sizing: border-box;
}

.button {
  background-color: var(--primary-color);
  color: white;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 600;
  transition: background-color 0.2s;
  width: 100%;
}

.button:hover:not(:disabled) {
  background-color: #0b5ed7;
}

.button:disabled {
  background-color: var(--secondary-color);
  cursor: not-allowed;
}

.metrics-container, .prediction-container {
  margin-top: 1.5rem;
  padding: 1rem;
  background-color: var(--bg-color);
  border-radius: 4px;
}

.metrics-container h3, .prediction-container h3 {
  margin-top: 0;
}

.metrics-table {
    width: 100%;
    border-collapse: collapse;
}
.metrics-table th, .metrics-table td {
    border: 1px solid var(--border-color);
    padding: 8px;
    text-align: left;
}
.metrics-table th {
    background-color: #e9ecef;
}

.prediction-result {
  font-size: 1.5rem;
  font-weight: bold;
  text-align: center;
  padding: 1rem;
  border-radius: 4px;
}

.prediction-result.exoplanet {
  background-color: #d1e7dd;
  color: var(--success-color);
}

.prediction-result.not-exoplanet {
  background-color: #f8d7da;
  color: var(--danger-color);
}

.error-message {
    margin-top: 1rem;
    color: var(--danger-color);
    background-color: #f8d7da;
    padding: 1rem;
    border-radius: 4px;
    border: 1px solid #f5c2c7;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 1.5rem;
  font-weight: bold;
  z-index: 10;
}

.card-content {
    position: relative;
}

frontend/src/App.jsx:
import React, { useState, useCallback } from 'react';

const API_BASE_URL = "[http://127.0.0.1:8000](http://127.0.0.1:8000)";

function App() {
  // --- STATE MANAGEMENT ---
  const [file, setFile] = useState(null);
  const [params, setParams] = useState({ n_estimators: 100, max_depth: '' });
  const [metrics, setMetrics] = useState(null);
  const [featureNames, setFeatureNames] = useState([]);
  const [predictValues, setPredictValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // --- API HANDLERS ---
  const handleTrain = async () => {
    if (!file) {
      setError("Please select a dataset file first.");
      return;
    }
    setIsLoading(true);
    setError(null);
    setMetrics(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append("file", file);
    
    // Construct query params for hyperparameters
    const hyperParams = new URLSearchParams({
        n_estimators: params.n_estimators,
        ...(params.max_depth && { max_depth: params.max_depth }) // Only include max_depth if it's set
    });

    try {
      const response = await fetch(`${API_BASE_URL}/upload_and_train?${hyperParams.toString()}`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "An error occurred during training.");
      }
      
      setMetrics(data.metrics);
      setFeatureNames(data.feature_names);
      setSessionId(data.session_id);
      // Initialize prediction state with empty strings for the new features
      setPredictValues(data.feature_names.reduce((acc, name) => ({ ...acc, [name]: '' }), {}));

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!sessionId) {
      setError("You must train a model before making a prediction.");
      return;
    }
    // Ensure all feature values are provided and are numbers
    const features = featureNames.map(name => parseFloat(predictValues[name]));
    if (features.some(isNaN)) {
        setError("Please fill in all feature fields with valid numbers.");
        return;
    }
    
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, features }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "An error occurred during prediction.");
      }
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // --- RENDER LOGIC ---
  const renderMetrics = () => {
    if (!metrics) return null;
    const { '0': notExoplanet, '1': exoplanet, accuracy } = metrics;

    return (
        <div className="metrics-container">
            <h3>Model Performance</h3>
            <table className="metrics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Not Exoplanet (0)</th>
                        <th>Exoplanet (1)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Precision</td><td>{notExoplanet.precision.toFixed(3)}</td><td>{exoplanet.precision.toFixed(3)}</td></tr>
                    <tr><td>Recall</td><td>{notExoplanet.recall.toFixed(3)}</td><td>{exoplanet.recall.toFixed(3)}</td></tr>
                    <tr><td>F1-Score</td><td>{notExoplanet['f1-score'].toFixed(3)}</td><td>{exoplanet['f1-score'].toFixed(3)}</td></tr>
                    <tr><td>Support</td><td>{notExoplanet.support}</td><td>{exoplanet.support}</td></tr>
                </tbody>
            </table>
            <h4>Overall Accuracy: {accuracy.toFixed(3)}</h4>
        </div>
    );
  };
  
  // --- COMPONENT RENDER ---
  return (
    <div className="App">
      <header className="header">
        <h1>Exoplanet Classifier</h1>
        <p>Train a model and predict new candidates using NASA's exoplanet data.</p>
      </header>
      
      {error && <div className="error-message">{error}</div>}

      <div className="card">
        <div className="card-content">
          {isLoading && <div className="loading-overlay">Processing...</div>}
          <h2>1. Train a New Model</h2>
          <div className="form-group">
            <label htmlFor="file-upload">Upload Exoplanet Dataset (CSV)</label>
            <input id="file-upload" type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} />
          </div>
          <div className="form-group">
            <label>Hyperparameters</label>
            <input type="number" placeholder="n_estimators (e.g., 100)" value={params.n_estimators} onChange={(e) => setParams({ ...params, n_estimators: parseInt(e.target.value) || 1 })}/>
            <input type="number" placeholder="max_depth (optional)" value={params.max_depth} onChange={(e) => setParams({ ...params, max_depth: e.target.value ? parseInt(e.target.value) : '' })} style={{marginTop: '10px'}}/>
          </div>
          <button className="button" onClick={handleTrain} disabled={isLoading}>
            Upload & Train
          </button>
          {renderMetrics()}
        </div>
      </div>

      {sessionId && (
        <div className="card">
            <div className="card-content">
                {isLoading && <div className="loading-overlay">Predicting...</div>}
                <h2>2. Predict a New Candidate</h2>
                {featureNames.map((name) => (
                    <div className="form-group" key={name}>
                        <label>{name}</label>
                        <input
                            type="text"
                            value={predictValues[name] || ''}
                            onChange={(e) => setPredictValues({ ...predictValues, [name]: e.target.value })}
                            placeholder={`Enter value for ${name}`}
                        />
                    </div>
                ))}
                <button className="button" onClick={handlePredict} disabled={isLoading}>
                    Predict
                </button>
                {prediction && (
                    <div className="prediction-container">
                        <h3>Prediction Result</h3>
                        <div className={`prediction-result ${prediction.prediction_label === 1 ? 'exoplanet' : 'not-exoplanet'}`}>
                            {prediction.prediction}
                        </div>
                    </div>
                )}
            </div>
        </div>
      )}
    </div>
  );
}

export default App;
```

---

## 🚀 How to Run the Project

### 1. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload --port 8000
```
The backend API will be running at `http://127.0.0.1:8000`.

### 2. Frontend Setup

```bash
# Open a NEW terminal and navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Run the React development server
npm run dev
```
The frontend application will be running at `http://127.0.0.1:5173`.

### 3. User Workflow

1.  **Open the web app** in your browser (`http://127.0.0.1:5173`).
2.  **Download Sample Data:** For testing, use the [Kaggle: Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) dataset (`cumulative.csv`).
3.  **Upload & Train:** Click "Choose File" and select `cumulative.csv`. Adjust hyperparameters if desired, then click "Upload & Train".
4.  **Review Metrics:** The UI will display the model's performance metrics.
5.  **Make a Prediction:** A new section will appear with input fields for each feature the model was trained on. Fill these in with data for a hypothetical candidate and click "Predict" to see the classification.
