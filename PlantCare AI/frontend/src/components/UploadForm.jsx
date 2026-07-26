import { useState, useRef } from "react";

export default function UploadForm({ onPredict, loading }) {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const inputRef = useRef(null);

  function handleFileChange(e) {
    const selected = e.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (file) onPredict(file);
  }

  return (
    <form className="upload-form" onSubmit={handleSubmit}>
      <div
        className="drop-zone"
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        {preview ? (
          <img src={preview} alt="Leaf preview" className="preview-img" />
        ) : (
          <p>Click to choose a leaf image (JPG, PNG, WEBP)</p>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleFileChange}
        hidden
      />
      <button type="submit" disabled={!file || loading}>
        {loading ? "Analyzing..." : "Classify Leaf"}
      </button>
    </form>
  );
}
