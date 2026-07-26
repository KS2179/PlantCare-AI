export default function PredictionResult({ result }) {
  if (!result) return null;
  const { predicted_class, confidence, top_k, info } = result;

  return (
    <div className={`result-card ${info.is_healthy ? "healthy" : "diseased"}`}>
      <div className="result-header">
        <h2>{info.disease || predicted_class}</h2>
        <span className="confidence-badge">{(confidence * 100).toFixed(1)}% confident</span>
      </div>
      {info.plant && <p className="plant-name">Plant: {info.plant}</p>}

      {!info.is_healthy && (
        <div className="info-grid">
          <InfoBlock label="Symptoms" text={info.symptoms} />
          <InfoBlock label="Causes" text={info.causes} />
          <InfoBlock label="Prevention" text={info.prevention} />
          <InfoBlock label="Treatment" text={info.treatment} />
        </div>
      )}

      {top_k?.length > 1 && (
        <div className="top-k">
          <h3>Other possibilities</h3>
          <ul>
            {top_k.slice(1).map((k) => (
              <li key={k.label}>
                {k.label} — {(k.confidence * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function InfoBlock({ label, text }) {
  if (!text) return null;
  return (
    <div className="info-block">
      <h4>{label}</h4>
      <p>{text}</p>
    </div>
  );
}
