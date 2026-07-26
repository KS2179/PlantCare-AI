import { useEffect, useState, useCallback } from "react";
import UploadForm from "./components/UploadForm.jsx";
import PredictionResult from "./components/PredictionResult.jsx";
import HistoryList from "./components/HistoryList.jsx";
import { predictImage, fetchHistory, fetchHistoryItem, deleteHistoryItem } from "./api.js";

export default function App() {
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadHistory = useCallback(async () => {
    try {
      const data = await fetchHistory();
      setHistory(data);
    } catch (e) {
      // History failing to load shouldn't block the rest of the UI
      console.error("Failed to load history", e);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  async function handlePredict(file) {
    setLoading(true);
    setError(null);
    try {
      const data = await predictImage(file);
      setResult(data);
      await loadHistory();
    } catch (e) {
      const detail = e?.response?.data?.detail;
      setError(detail || "Something went wrong classifying this image.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectHistory(id) {
    try {
      const data = await fetchHistoryItem(id);
      setResult(data);
    } catch (e) {
      console.error("Failed to load history item", e);
    }
  }

  async function handleDeleteHistory(id) {
    try {
      await deleteHistoryItem(id);
      await loadHistory();
    } catch (e) {
      console.error("Failed to delete history item", e);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>🌿 Plant Disease Classifier</h1>
        <p>Upload a photo of a leaf to identify diseases and get treatment guidance.</p>
      </header>

      <main>
        <section className="left-panel">
          <UploadForm onPredict={handlePredict} loading={loading} />
          {error && <div className="error-banner">{error}</div>}
          <PredictionResult result={result} />
        </section>

        <aside className="right-panel">
          <h3>Prediction History</h3>
          <HistoryList items={history} onSelect={handleSelectHistory} onDelete={handleDeleteHistory} />
        </aside>
      </main>
    </div>
  );
}
