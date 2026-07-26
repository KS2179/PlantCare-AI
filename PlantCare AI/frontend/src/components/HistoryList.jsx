export default function HistoryList({ items, onSelect, onDelete }) {
  if (!items?.length) {
    return <p className="empty-state">No predictions yet -- upload a leaf image to get started.</p>;
  }

  return (
    <ul className="history-list">
      {items.map((item) => (
        <li key={item.id} className={item.is_healthy ? "healthy" : "diseased"}>
          <button className="history-item-btn" onClick={() => onSelect(item.id)}>
            <span className="history-disease">{item.disease || item.predicted_class}</span>
            <span className="history-meta">
              {item.plant} · {(item.confidence * 100).toFixed(1)}% ·{" "}
              {new Date(item.created_at).toLocaleString()}
            </span>
          </button>
          <button className="delete-btn" onClick={() => onDelete(item.id)} aria-label="Delete">
            ×
          </button>
        </li>
      ))}
    </ul>
  );
}
