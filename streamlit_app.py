import os, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =========================
# Utilidades de carga/cache
# =========================
@st.cache_resource(show_spinner=False)
def _latest_art_dir(root="artefactos") -> Path:
    subs = [Path(p) for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
    if not subs:
        raise FileNotFoundError(f"No hay versiones en '{root}'. Exporta artefactos con el Paso 11.")
    subs.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return subs[0]

@st.cache_resource(show_spinner=True)
def load_artifacts(artefacts_root="artefactos"):
    art_dir = _latest_art_dir(artefacts_root)

    with open(art_dir / "input_schema.json", "r", encoding="utf-8") as f:
        input_schema = json.load(f)
    with open(art_dir / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(art_dir / "decision_policy.json", "r", encoding="utf-8") as f:
        policy = json.load(f)

    # Para deserializar pipelines con SMOTE
    try:
        import imblearn  # noqa: F401
    except Exception:
        pass

    winner = policy["winner"]
    threshold = float(policy.get("threshold", 0.5))
    pipe = joblib.load(art_dir / f"pipeline_{winner}.joblib")

    # Compatibilidad de schema
    if "columns" in input_schema and "dtypes" in input_schema:
        features = input_schema["columns"]
        dtypes   = input_schema["dtypes"]
    else:  # legado: {col: dtype}
        features = list(input_schema.keys())
        dtypes   = input_schema

    # Samples (opcional)
    samples = None
    spath = art_dir / "sample_inputs.json"
    if spath.exists():
        with open(spath, "r", encoding="utf-8") as f:
            samples = json.load(f)

    return {
        "art_dir": art_dir,
        "pipe": pipe,
        "features": features,
        "dtypes": dtypes,
        "label_map": label_map,
        "policy": policy,
        "threshold": threshold,
        "samples": samples,
        "winner": winner
    }

def coerce_and_align(df: pd.DataFrame, *, features, dtypes) -> pd.DataFrame:
    df = df.copy()
    # Crear columnas faltantes
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
    # Coaccionar tipos
    for c in features:
        t = str(dtypes.get(c, "object")).lower()
        if t.startswith(("int", "float")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif t in ("bool", "boolean"):
            df[c] = (df[c].astype("string").str.strip().str.lower()
                     .map({"true": True, "false": False}))
            df[c] = df[c].fillna(False).astype(bool)
        else:
            df[c] = df[c].astype("string").str.strip()
    return df[features]

def predict_proba(pipe, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1]

def predict_with_threshold(pipe, X: pd.DataFrame, thr: float) -> pd.DataFrame:
    proba = predict_proba(pipe, X)
    yhat  = (proba >= float(thr)).astype(int)
    return pd.DataFrame({"proba_ALTO": proba, "pred_num": yhat})

# ==========
# Interfaz UI
# ==========
st.set_page_config(page_title="Modelo de Clasificación — Inferencia", page_icon="🤖", layout="wide")
st.title("🤖 Clasificación — Inferencia con artefactos")
st.caption("Carga automática de pipeline + schema + policy (Paso 11). Predicción individual y por lotes.")

# Cargar artefactos
try:
    A = load_artifacts("artefactos")
except Exception as e:
    st.error(f"No se pudieron cargar artefactos: {e}")
    st.stop()

st.success(f"Versión en uso: **{A['art_dir'].name}** — Modelo ganador: **{A['winner']}**")
with st.expander("Ver detalles del esquema de entrada", expanded=False):
    dd = pd.DataFrame(
        {"column": A["features"], "dtype": [A["dtypes"][c] for c in A["features"]]}
    )
    st.dataframe(dd, hide_index=True, use_container_width=True)

# Sidebar: threshold y descarga de plantilla
st.sidebar.header("Ajustes")
thr = st.sidebar.slider("Umbral de decisión (ALTO=1)", 0.0, 1.0, float(A["threshold"]), 0.01)
st.sidebar.caption(f"Umbral por defecto del policy: {A['threshold']:.2f}")

# Plantilla CSV
template_df = pd.DataFrame({c: ["" ] for c in A["features"]})
st.sidebar.download_button(
    "Descargar plantilla CSV",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="template_inferencia.csv",
    mime="text/csv"
)

tab1, tab2 = st.tabs(["🔹 Predicción individual", "📄 Predicción por lote (CSV)"])

# -------------------------
# Predicción individual (UI)
# -------------------------
with tab1:
    st.subheader("🔹 Predicción individual")
    col_left, col_right = st.columns(2)

    # Sugerencias desde samples si existen
    sample_hint = (A["samples"][0] if A["samples"] else None)

    user_input = {}
    for i, col in enumerate(A["features"]):
        t = str(A["dtypes"].get(col, "object")).lower()
        container = col_left if i % 2 == 0 else col_right

        with container:
            if t.startswith(("int", "float")):
                default = None
                if sample_hint and col in sample_hint:
                    try:
                        default = float(sample_hint[col])
                    except Exception:
                        default = None
                user_input[col] = st.number_input(col, value=default, step=1.0 if t.startswith("int") else 0.01, format="%.4f")
            elif t in ("bool", "boolean"):
                default = False
                if sample_hint and col in sample_hint:
                    default = bool(sample_hint[col])
                user_input[col] = st.checkbox(col, value=default)
            else:
                default = ""
                if sample_hint and col in sample_hint:
                    default = str(sample_hint[col])
                user_input[col] = st.text_input(col, value=default)

    if st.button("Predecir", use_container_width=True):
        X1 = coerce_and_align(pd.DataFrame([user_input]), features=A["features"], dtypes=A["dtypes"])
        out = predict_with_threshold(A["pipe"], X1, thr)
        inv = {v: k for k, v in A["label_map"].items()}
        y_label = out["pred_num"].iloc[0]
        st.metric("Probabilidad ALTO=1", f"{out['proba_ALTO'].iloc[0]:.3f}")
        st.metric("Decisión", f"{inv[int(y_label)]} ({int(y_label)})")

# -------------------------
# Predicción por lote (CSV)
# -------------------------
with tab2:
    st.subheader("📄 Predicción por lote (CSV)")
    up = st.file_uploader("Sube un CSV con las columnas del schema", type=["csv"])

    if up is not None:
        try:
            df_in = pd.read_csv(up)
        except Exception:
            df_in = pd.read_csv(up, encoding="latin-1")

        st.write("Vista previa:")
        st.dataframe(df_in.head(), use_container_width=True)

        Xb = coerce_and_align(df_in, features=A["features"], dtypes=A["dtypes"])
        out = predict_with_threshold(A["pipe"], Xb, thr)
        inv = {v: k for k, v in A["label_map"].items()}
        out["pred_label"] = out["pred_num"].map(lambda x: inv[int(x)])

        result = pd.concat([df_in.reset_index(drop=True), out], axis=1)

        st.success(f"Predicciones listas (n={len(result)})")
        st.dataframe(result.head(50), use_container_width=True)

        st.download_button(
            "Descargar resultados CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="predicciones.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("---")
st.caption("© Tu equipo — App generada a partir de artefactos del pipeline (Paso 11).")
