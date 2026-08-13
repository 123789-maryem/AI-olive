import { useState, useEffect } from "react";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [tab, setTab] = useState("analyse");

  const fetchHistory = async () => {
    const res = await fetch("http://localhost:5000/history");
    const data = await res.json();
    setHistory(data);
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!image) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("image", image);

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
      fetchHistory();
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div style={{ textAlign: "center", padding: "2rem", maxWidth: 800, margin: "0 auto" }}>
      <h1>🫒 Olive Disease AI</h1>

      <div style={{ marginBottom: 20 }}>
        <button onClick={() => setTab("analyse")} style={{ marginRight: 10 }}>
          Analyser
        </button>
        <button onClick={() => setTab("gallery")}>Galerie / Historique</button>
      </div>

      {tab === "analyse" && (
        <div>
          <input type="file" accept="image/*" onChange={handleImageChange} />
          {preview && (
            <img src={preview} alt="preview" style={{ width: 300, marginTop: 20, borderRadius: 8 }} />
          )}
          <br />
          <button onClick={handleSubmit} disabled={!image || loading} style={{ marginTop: 20 }}>
            {loading ? "Analyse en cours..." : "Analyser"}
          </button>

          {result && (
            <div
              style={{
                marginTop: 20,
                textAlign: "left",
                maxWidth: 500,
                margin: "20px auto",
                padding: 16,
                border: "1px solid #444",
                borderRadius: 8,
              }}
            >
              <h2 style={{ textAlign: "center" }}>{result.disease}</h2>
              <p style={{ textAlign: "center", opacity: 0.7 }}>
                Confiance: {(result.confidence * 100).toFixed(1)}%
              </p>
              <hr />
              <p><strong>📋 Description :</strong></p>
              <p>{result.description}</p>
              <p><strong>💊 Traitement recommandé :</strong></p>
              <p>{result.treatment}</p>
              <p style={{ fontSize: 12, opacity: 0.6, marginTop: 10 }}>
                ⚠️ Ces informations sont indicatives. Consultez un expert agricole pour un diagnostic précis.
              </p>
            </div>
          )}
        </div>
      )}

      {tab === "gallery" && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
            gap: 16,
          }}
        >
          {history.length === 0 && <p>Aucune analyse pour le moment.</p>}
          {history.map((item) => (
            <div key={item.id} style={{ border: "1px solid #444", borderRadius: 8, padding: 10 }}>
              <img
                src={`http://localhost:5000${item.image_url}`}
                alt={item.disease}
                style={{ width: "100%", borderRadius: 6 }}
              />
              <p style={{ fontWeight: "bold", margin: "8px 0 2px" }}>{item.disease}</p>
              <p style={{ fontSize: 12, opacity: 0.7 }}>
                {(item.confidence * 100).toFixed(1)}% — {new Date(item.date).toLocaleString()}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;