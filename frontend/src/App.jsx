import React, { useState, useCallback } from 'react';

const API_BASE_URL = "https://exoplanet-api-byqb.onrender.com";

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
