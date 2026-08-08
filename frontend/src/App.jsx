import React, { useState } from 'react';
import { 
  Sparkles, 
  Orbit, 
  UploadCloud, 
  Cpu, 
  Award, 
  Globe, 
  AlertTriangle, 
  CheckCircle2, 
  XCircle, 
  RefreshCw, 
  Layers,
  FileSpreadsheet
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const SAMPLE_CSV_CONTENT = `koi_disposition,koi_period,koi_duration,koi_depth,koi_prad,koi_teq
CONFIRMED,9.259,2.2,615.8,2.26,793
FALSE POSITIVE,10.98,4.5,874.8,33.9,1395
CONFIRMED,4.55,1.8,120.8,1.06,925
FALSE POSITIVE,19.89,1.7,10829,14.6,638
CONFIRMED,1.77,2.4,86.1,0.8,1406
CONFIRMED,3.5225,3.19,114.6,1.48,1168
FALSE POSITIVE,1.14,0.7,91.2,0.9,2100
CONFIRMED,2.88,5.6,210.9,1.59,1201
CONFIRMED,7.33,2.4,130.4,1.1,830
FALSE POSITIVE,0.78,1.2,110.6,1.11,2218
CONFIRMED,11.3,3.9,400.0,2.1,900
FALSE POSITIVE,22.1,5.2,12000.0,15.0,550
CONFIRMED,4.98,2.8,350.0,1.9,1050
FALSE POSITIVE,0.52,1.3,121.0,1.2,2500
CONFIRMED,6.8,3.5,250.0,1.7,950
CONFIRMED,9.5,2.1,150.0,1.3,800
FALSE POSITIVE,15.2,4.8,9000.0,12.0,600
CONFIRMED,3.1,2.9,180.0,1.4,1100
FALSE POSITIVE,30.5,6.1,15000.0,20.0,500
CONFIRMED,2.5,1.9,100.0,1.0,1300
CONFIRMED,5.6,3.2,280.0,1.8,1000
FALSE POSITIVE,1.9,0.9,150.0,1.5,1900
CONFIRMED,8.2,4.1,320.0,2.0,920
FALSE POSITIVE,0.9,1.1,130.0,1.3,2300
CONFIRMED,12.9,5.5,450.0,2.3,850`;

const FEATURE_HELP = {
  koi_period: "Orbital Period (days) - Time to complete 1 orbit",
  koi_duration: "Transit Duration (hours) - Duration brightness dips",
  koi_depth: "Transit Depth (ppm) - Starlight fraction blocked",
  koi_prad: "Planetary Radius (Earth radii) - Estimated radius",
  koi_teq: "Equilibrium Temp (K) - Estimated planetary temp"
};

const PRESETS = {
  earth_like: {
    label: "🌍 Earth-like Candidate",
    values: { koi_period: 3.5225, koi_duration: 3.19, koi_depth: 114.6, koi_prad: 1.48, koi_teq: 1168 }
  },
  gas_giant: {
    label: "🪐 Hot Jupiter Candidate",
    values: { koi_period: 9.259, koi_duration: 2.2, koi_depth: 615.8, koi_prad: 2.26, koi_teq: 793 }
  },
  false_positive: {
    label: "☄️ False Positive Binary",
    values: { koi_period: 10.98, koi_duration: 4.5, koi_depth: 874.8, koi_prad: 33.9, koi_teq: 1395 }
  }
};

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [params, setParams] = useState({ n_estimators: 100, max_depth: '' });
  const [metrics, setMetrics] = useState(null);
  const [featureNames, setFeatureNames] = useState([]);
  const [predictValues, setPredictValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("Processing...");
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setFileName(selectedFile.name);
    }
  };

  const loadDemoDataset = () => {
    const blob = new Blob([SAMPLE_CSV_CONTENT], { type: 'text/csv' });
    const demoFile = new File([blob], 'demo_exoplanet_dataset.csv', { type: 'text/csv' });
    setFile(demoFile);
    setFileName("demo_exoplanet_dataset.csv (Ready to train)");
    setError(null);
  };

  const handleTrain = async (customFile = null) => {
    const fileToUpload = customFile || file;
    if (!fileToUpload) {
      setError("Please select a dataset CSV file or use the Demo Dataset button.");
      return;
    }

    setIsLoading(true);
    setLoadingText("Training Random Forest Classifier on NASA Kepler dataset...");
    setError(null);
    setMetrics(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append("file", fileToUpload);
    
    const hyperParams = new URLSearchParams({
      n_estimators: params.n_estimators,
      ...(params.max_depth && { max_depth: params.max_depth })
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

      // Pre-fill prediction values with default Earth-like candidate
      setPredictValues(PRESETS.earth_like.values);

    } catch (err) {
      setError(err.message || "Failed to communicate with API server.");
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!sessionId) {
      setError("You must train a model before making a prediction.");
      return;
    }

    const features = featureNames.map(name => parseFloat(predictValues[name]));
    if (features.some(isNaN)) {
      setError("Please fill in all candidate feature values with valid numbers.");
      return;
    }
    
    setIsLoading(true);
    setLoadingText("Analyzing candidate celestial parameters...");
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
      setError(err.message || "Failed to calculate prediction.");
    } finally {
      setIsLoading(false);
    }
  };

  const applyPreset = (presetKey) => {
    if (PRESETS[presetKey]) {
      setPredictValues(PRESETS[presetKey].values);
    }
  };

  const renderMetrics = () => {
    if (!metrics) return null;
    const { '0': notExoplanet, '1': exoplanet, accuracy } = metrics;

    return (
      <div className="metrics-container">
        <div className="accuracy-hero">
          <div>
            <div className="accuracy-title">Model Overall Accuracy</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Random Forest Ensemble Evaluation</div>
          </div>
          <div className="accuracy-value">{(accuracy * 100).toFixed(1)}%</div>
        </div>

        <div className="metrics-table-wrapper">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Classification Metric</th>
                <th>Not Exoplanet (0)</th>
                <th>Confirmed Exoplanet (1)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Precision</strong></td>
                <td>{notExoplanet ? notExoplanet.precision.toFixed(3) : 'N/A'}</td>
                <td>{exoplanet ? exoplanet.precision.toFixed(3) : 'N/A'}</td>
              </tr>
              <tr>
                <td><strong>Recall</strong></td>
                <td>{notExoplanet ? notExoplanet.recall.toFixed(3) : 'N/A'}</td>
                <td>{exoplanet ? exoplanet.recall.toFixed(3) : 'N/A'}</td>
              </tr>
              <tr>
                <td><strong>F1-Score</strong></td>
                <td>{notExoplanet ? notExoplanet['f1-score'].toFixed(3) : 'N/A'}</td>
                <td>{exoplanet ? exoplanet['f1-score'].toFixed(3) : 'N/A'}</td>
              </tr>
              <tr>
                <td><strong>Sample Support</strong></td>
                <td>{notExoplanet ? notExoplanet.support : 0}</td>
                <td>{exoplanet ? exoplanet.support : 0}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    );
  };
  
  return (
    <div className="App">
      <header className="header">
        <div className="header-badge">
          <Orbit size={16} /> NASA Space Apps Challenge Blueprint
        </div>
        <h1>Exoplanet Classifier AI</h1>
        <p>
          Machine learning system designed to discover extra-solar planet candidates using NASA Kepler space telescope transit light-curve features.
        </p>
      </header>
      
      {error && (
        <div className="error-message">
          <AlertTriangle size={20} />
          <span>{error}</span>
        </div>
      )}

      <div className="card">
        <div className="card-content">
          {isLoading && (
            <div className="loading-overlay">
              <div className="spinner"></div>
              <div>{loadingText}</div>
            </div>
          )}

          <div className="card-title-row">
            <h2><Cpu size={24} /> 1. Train Model Engine</h2>
            <span className="step-indicator">Step 1 of 2</span>
          </div>

          <div className="form-group">
            <label>Upload Exoplanet Dataset (CSV)</label>
            <label htmlFor="file-upload" className="file-dropzone">
              <UploadCloud size={36} />
              {fileName ? (
                <span className="file-name-display">{fileName}</span>
              ) : (
                <span>Click or Drag & Drop NASA Kepler CSV dataset</span>
              )}
              <span className="field-help">Required columns: koi_disposition, koi_period, koi_duration, koi_depth, koi_prad, koi_teq</span>
            </label>
            <input id="file-upload" type="file" accept=".csv" onChange={handleFileChange} />
          </div>

          <div className="form-group">
            <label>Random Forest Hyperparameters</label>
            <div className="input-row">
              <div>
                <span className="field-help" style={{ display: 'block', marginBottom: '4px' }}>n_estimators (Trees)</span>
                <input 
                  type="number" 
                  placeholder="e.g., 100" 
                  value={params.n_estimators} 
                  onChange={(e) => setParams({ ...params, n_estimators: parseInt(e.target.value) || 1 })}
                />
              </div>
              <div>
                <span className="field-help" style={{ display: 'block', marginBottom: '4px' }}>max_depth (Optional)</span>
                <input 
                  type="number" 
                  placeholder="e.g., 10 (Leave blank for max)" 
                  value={params.max_depth} 
                  onChange={(e) => setParams({ ...params, max_depth: e.target.value ? parseInt(e.target.value) : '' })}
                />
              </div>
            </div>
          </div>

          <div className="btn-group">
            <button className="button" onClick={() => handleTrain()} disabled={isLoading}>
              <Sparkles size={18} /> Upload & Train Model
            </button>
            <button className="button button-secondary" onClick={() => { loadDemoDataset(); }} disabled={isLoading}>
              <FileSpreadsheet size={18} /> Load Demo NASA Dataset
            </button>
          </div>

          {renderMetrics()}
        </div>
      </div>

      {sessionId && (
        <div className="card" style={{ animation: 'fadeIn 0.5s ease-out' }}>
          <div className="card-content">
            {isLoading && (
              <div className="loading-overlay">
                <div className="spinner"></div>
                <div>{loadingText}</div>
              </div>
            )}

            <div className="card-title-row">
              <h2><Globe size={24} /> 2. Predict New Candidate</h2>
              <div className="session-info">
                <Layers size={14} /> Session: {sessionId.substring(0, 8)}...
              </div>
            </div>

            <div className="preset-section">
              <div className="preset-title">Quick Test Presets:</div>
              <div className="preset-grid">
                {Object.keys(PRESETS).map((key) => (
                  <button key={key} type="button" className="preset-btn" onClick={() => applyPreset(key)}>
                    {PRESETS[key].label}
                  </button>
                ))}
              </div>
            </div>

            <div className="input-row" style={{ marginBottom: '1.5rem' }}>
              {featureNames.map((name) => (
                <div className="form-group" key={name} style={{ marginBottom: '0.5rem' }}>
                  <label htmlFor={`input-${name}`}>
                    {name}
                  </label>
                  <span className="field-help" style={{ display: 'block', marginBottom: '6px' }}>
                    {FEATURE_HELP[name] || 'Transit feature parameter'}
                  </span>
                  <input
                    id={`input-${name}`}
                    type="text"
                    value={predictValues[name] !== undefined ? predictValues[name] : ''}
                    onChange={(e) => setPredictValues({ ...predictValues, [name]: e.target.value })}
                    placeholder={`Enter ${name}`}
                  />
                </div>
              ))}
            </div>

            <button className="button" onClick={handlePredict} disabled={isLoading}>
              <Orbit size={18} /> Run Classification Inference
            </button>

            {prediction && (
              <div className="prediction-container">
                <div className={`prediction-result ${prediction.prediction_label === 1 ? 'exoplanet' : 'not-exoplanet'}`}>
                  <div className="result-icon-wrapper">
                    {prediction.prediction_label === 1 ? (
                      <CheckCircle2 size={36} />
                    ) : (
                      <XCircle size={36} />
                    )}
                  </div>
                  <div className="prediction-label-text">
                    {prediction.prediction}
                  </div>
                  <div className="prediction-desc">
                    {prediction.prediction_label === 1 ? (
                      "Target exhibits transit signatures highly consistent with a confirmed exoplanet."
                    ) : (
                      "Target metrics match signatures of a false positive signal or binary star transit."
                    )}
                  </div>
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

