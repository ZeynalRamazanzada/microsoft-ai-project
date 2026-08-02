import { useEffect, useMemo, useState } from "react";
import {
  fetchSchema,
  fetchExample,
  predict,
  health,
  SchemaField,
  PredictResponse,
  FeatureValues,
} from "./api";

const GROUP_ORDER = [
  "Reaction Time",
  "Choice Behavior",
  "DDM Parameters",
  "Session / Context",
];

export default function App() {
  const [schema, setSchema] = useState<SchemaField[]>([]);
  const [values, setValues] = useState<FeatureValues>({});
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  useEffect(() => {
    (async () => {
      try {
        await health();
        setApiUp(true);
      } catch {
        setApiUp(false);
      }
      try {
        const s = await fetchSchema();
        setSchema(s.fields);
        const init: FeatureValues = {};
        s.fields.forEach((f) => (init[f.name] = f.example));
        setValues(init);
      } catch (e) {
        setError("Failed to load schema from API. Is the backend running on :8000?");
      }
    })();
  }, []);

  const groupedFields = useMemo(() => {
    const map = new Map<string, SchemaField[]>();
    for (const f of schema) {
      if (!map.has(f.group)) map.set(f.group, []);
      map.get(f.group)!.push(f);
    }
    return GROUP_ORDER.map((g) => [g, map.get(g) ?? []] as const).filter(
      ([, fs]) => fs.length > 0
    );
  }, [schema]);

  function updateValue(name: string, raw: string) {
    const num = raw === "" ? NaN : Number(raw);
    setValues((prev) => ({ ...prev, [name]: num }));
  }

  async function loadExample(label: "fatigued" | "not_fatigued") {
    setError(null);
    try {
      const ex = await fetchExample(label);
      setValues(ex);
    } catch {
      setError(`Failed to load ${label} example.`);
    }
  }

  async function handlePredict() {
    setError(null);
    setLoading(true);
    try {
      const invalid = Object.entries(values).find(([, v]) => Number.isNaN(v));
      if (invalid) {
        setError(`Field "${invalid[0]}" is empty or invalid.`);
        setLoading(false);
        return;
      }
      const res = await predict(values);
      setResult(res);
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || "Prediction failed.";
      setError(typeof msg === "string" ? msg : JSON.stringify(msg));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🧠 BDFS — Behavioral Decision Fatigue Scoring</h1>
        <p className="subtitle">
          Single-trial fatigue prediction · XGBoost · F1 = 0.866 · ROC-AUC = 0.967
        </p>
        <hr />
      </header>

      <div className="health">
        <span className={`dot ${apiUp ? "ok" : apiUp === false ? "bad" : ""}`} />
        {apiUp === null
          ? "Connecting to API…"
          : apiUp
          ? "Backend connected at /api → :8000"
          : "Backend not reachable. Start uvicorn on port 8000."}
      </div>

      <div className="layout">
        <section className="panel">
          <h2>Input features</h2>
          <div className="toolbar">
            <button className="danger" onClick={() => loadExample("fatigued")}>
              Load fatigued example
            </button>
            <button className="success" onClick={() => loadExample("not_fatigued")}>
              Load not-fatigued example
            </button>
          </div>

          {groupedFields.map(([group, fields]) => (
            <div key={group} className="group">
              <h3>{group}</h3>
              <div className="group-grid">
                {fields.map((f) => (
                  <div key={f.name} className="field">
                    <label htmlFor={f.name}>{f.name}</label>
                    <input
                      id={f.name}
                      type="number"
                      step={f.type === "int" ? 1 : "any"}
                      min={f.min}
                      max={f.max}
                      value={Number.isNaN(values[f.name]) ? "" : values[f.name] ?? ""}
                      onChange={(e) => updateValue(f.name, e.target.value)}
                    />
                  </div>
                ))}
              </div>
            </div>
          ))}

          <button
            className="primary"
            onClick={handlePredict}
            disabled={loading || schema.length === 0}
          >
            {loading && <span className="spinner" />}
            {loading ? "Predicting…" : "Predict fatigue"}
          </button>
        </section>

        <section className="panel">
          <h2>Prediction</h2>
          {error && <div className="error">{error}</div>}
          {result ? <ResultView r={result} /> : !error && <Placeholder />}
        </section>
      </div>
    </div>
  );
}

function ResultView({ r }: { r: PredictResponse }) {
  const pct = Math.round(r.probability * 100);
  return (
    <>
      <div className="result">
        <span className={`badge ${r.label}`}>
          {r.label === "fatigued" ? "FATIGUED" : "NOT FATIGUED"}
        </span>
        <div className="prob-value">{pct}%</div>
        <div className="prob-label">probability of fatigue</div>
        <div className="gauge">
          <div className="gauge-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="prob-label">
          P(not fatigued) = {r.probability_not_fatigued.toFixed(4)} &nbsp;·&nbsp;
          model = {r.model_used}
        </div>
      </div>

      {r.top_contributing_features.length > 0 && (
        <div className="contrib">
          <h3>Top contributing features (by global SHAP)</h3>
          {r.top_contributing_features.map((c) => (
            <div key={c.feature} className="contrib-row">
              <span>{c.feature}</span>
              <span>{c.value.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function Placeholder() {
  return (
    <div className="placeholder">
      Fill the form (or click <strong>Load example</strong>) and press{" "}
      <strong>Predict fatigue</strong>.
    </div>
  );
}
