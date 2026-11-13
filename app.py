"""
app.py — Interfaz Streamlit para el monitoreo de fraude transaccional.

Conecta la entrada del usuario (formulario o CSV) con scoring.py:

    entrada -> validación -> score_transaccion / evaluar_riesgo -> decisión -> pantalla

Ejecutar localmente:
    streamlit run app.py
"""

import io
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring
from scoring import (
    ArtefactosNoEncontradosError,
    EntradaInvalidaError,
    evaluar_riesgo,
)

st.set_page_config(page_title="Monitoreo de fraude", layout="wide")

# ---------------------------------------------------------------------------
# Estilo: tipografía IBM Plex y paleta propia (colores base en .streamlit/config.toml)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp input,
    .stApp textarea, .stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
    [data-testid="stCaptionContainer"], [data-baseweb="select"] div {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .stApp h1, .stApp h2, .stApp h3 { color: #13233a; letter-spacing: -0.01em; }
    .stApp h3 { font-weight: 600; font-size: 1.15rem; margin-top: .6rem;
                padding-bottom: .35rem; border-bottom: 1px solid #ddd6c8; }
    [data-testid="stSidebar"] { border-right: 1px solid #ddd6c8; }
    [data-testid="stSidebar"] h2 { font-size: 1rem; text-transform: uppercase;
                                   letter-spacing: .08em; color: #5b6472; }
    .stApp button[kind="primary"] { font-weight: 600; letter-spacing: .02em;
                                    padding: .55rem 1.6rem; border-radius: 6px; }
    .encabezado .kicker { font-size: .78rem; font-weight: 600; letter-spacing: .14em;
                          text-transform: uppercase; color: #0e5a5f; margin-bottom: .2rem; }
    .encabezado .titulo { font-size: 2.1rem; font-weight: 700; color: #13233a;
                          line-height: 1.15; margin: 0 0 .5rem 0; }
    .encabezado .sub { color: #4a5363; max-width: 760px; font-size: 1rem; margin-bottom: 1.4rem; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stFileUploaderDropzone"] svg,
    [data-testid="stFileUploaderFile"] > div:first-child { display: none; }
    .num { font-family: 'IBM Plex Mono', monospace !important; font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Registro de uso: una línea JSON por predicción o error. Va a stdout (pestaña
# "Logs" del Space en Hugging Face) y a logs/uso.jsonl cuando se corre local.
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_PATH = os.path.join(LOG_DIR, "uso.jsonl")
logger = logging.getLogger("monitoreo_fraude")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)


def registrar_uso(evento: str, **datos):
    """Escribe un evento de uso (sin datos personales) en stdout y en logs/uso.jsonl."""
    linea = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "evento": evento, **datos}, ensure_ascii=False)
    logger.info(linea)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(linea + "\n")
    except OSError:
        pass  # en el Space el disco puede ser de solo lectura; stdout basta

# ---------------------------------------------------------------------------
# Constantes de la interfaz
# ---------------------------------------------------------------------------
CAMPOS_TX = ["monto", "hora", "canal", "categoria", "es_nuevo", "horas_desde_ultima"]
MAX_TX = scoring.W_COMPLETO  # 10

COLORES = {
    "verde": ("#2e7d4f", "#e7f1ea", "Riesgo bajo"),
    "ámbar": ("#b7791f", "#f7eedb", "Riesgo intermedio, vigilar"),
    "rojo": ("#a63232", "#f5e3e1", "Riesgo alto"),
}

# Campos agregados que el usuario ingresa (los demás se derivan de la última tx).
AGG_MANUALES = {
    "n_tx_1h": ("Transacciones en la última hora", 0, 1, "int"),
    "n_tx_24h": ("Transacciones en las últimas 24 h", 0, 1, "int"),
    "div_comercio_7d": ("Categorías distintas en 7 días", 0, 2, "int"),
    "div_canal_7d": ("Canales distintos en 7 días", 0, 1, "int"),
    "ratio_monto_hist": ("Monto / promedio histórico", 0.0, 1.0, "float"),
    "n_tx_historicas": ("Transacciones históricas del cliente", 0, 24, "int"),
    "monto_prom_24h": ("Monto promedio 24 h (Q)", 0.0, 88.0, "float"),
    "monto_max_dia": ("Monto máximo del día (Q)", 0.0, 88.0, "float"),
}


# ---------------------------------------------------------------------------
# Ejemplos precargados
# ---------------------------------------------------------------------------
def _historia_legitima():
    base = [
        ("POS", "comida", 85, 14, 0, 60), ("POS", "retail", 110, 16, 0, 72),
        ("POS", "comida", 70, 13, 0, 55), ("POS", "comida", 95, 15, 0, 80),
        ("online", "retail", 120, 17, 0, 66), ("POS", "comida", 80, 14, 0, 90),
        ("POS", "servicios", 100, 15, 0, 48), ("POS", "comida", 90, 13, 0, 70),
        ("POS", "retail", 105, 16, 0, 62),
    ]
    return base


def _df(filas):
    return pd.DataFrame(
        [dict(zip(["canal", "categoria", "monto", "hora", "es_nuevo", "horas_desde_ultima"], f))
         for f in filas]
    )[CAMPOS_TX]


def _agg_base(**kw):
    d = dict(n_tx_1h=1, n_tx_24h=1, div_comercio_7d=2, div_canal_7d=1, ratio_monto_hist=1.0,
             n_tx_historicas=24, es_fin_semana=0, es_comercio_nuevo=0,
             monto_prom_24h=88.0, monto_max_dia=88.0)
    d.update(kw)
    return d


def ejemplos():
    legit = _historia_legitima() + [("POS", "comida", 88, 14, 0, 75)]
    fraude = _historia_legitima()[-5:] + [
        ("online", "retail", 12, 3, 1, 60), ("online", "entretenimiento", 25, 3, 1, 0.05),
        ("online", "servicios", 30, 3, 1, 0.07), ("online", "viajes", 18, 3, 1, 0.05),
        ("online", "retail", 1400, 3, 1, 0.1),
    ]
    limite = _historia_legitima() + [("POS", "comida", 95, 14, 0, 60)]
    return {
        "Transacción legítima": (
            _df(legit), _agg_base(ratio_monto_hist=0.98)),
        "Transacción fraudulenta (escalada rápida)": (
            _df(fraude), _agg_base(n_tx_1h=5, n_tx_24h=5, div_comercio_7d=4, div_canal_7d=2,
                                   ratio_monto_hist=15.0, es_comercio_nuevo=1,
                                   monto_prom_24h=297.0, monto_max_dia=1400.0)),
        "Caso límite (zona ámbar)": (
            _df(limite), _agg_base(monto_prom_24h=95.0, monto_max_dia=95.0)),
        "En blanco": (
            pd.DataFrame([{"monto": 0.0, "hora": 12.0, "canal": "POS", "categoria": "comida",
                           "es_nuevo": 0, "horas_desde_ultima": 24.0}]), _agg_base(
                n_tx_1h=0, n_tx_24h=0, div_comercio_7d=0, div_canal_7d=0, ratio_monto_hist=1.0,
                n_tx_historicas=0, monto_prom_24h=0.0, monto_max_dia=0.0)),
    }


# ---------------------------------------------------------------------------
# Carga de artefactos (una sola vez) y utilidades
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo():
    """Fuerza la carga de artefactos de scoring.py y devuelve (AGG_COLS, umbral)."""
    scoring._cargar_artefactos()
    return list(scoring.AGG_COLS), float(scoring._UMBRAL)


def construir_agregadas(ultima_tx: dict, manuales: dict) -> dict:
    """Arma las 13 variables agregadas en el formato que espera scoring.py."""
    return {
        "n_tx_1h": manuales["n_tx_1h"],
        "n_tx_24h": manuales["n_tx_24h"],
        "div_comercio_7d": manuales["div_comercio_7d"],
        "div_canal_7d": manuales["div_canal_7d"],
        "ratio_monto_hist": manuales["ratio_monto_hist"],
        "n_tx_historicas": manuales["n_tx_historicas"],
        "hora": float(ultima_tx["hora"]),
        "es_fin_semana": int(manuales["es_fin_semana"]),
        "es_comercio_nuevo": int(manuales["es_comercio_nuevo"]),
        "log_monto": float(np.log1p(float(ultima_tx["monto"]))),
        "log_monto_prom_24h": float(np.log1p(float(manuales["monto_prom_24h"]))),
        "log_monto_max_dia": float(np.log1p(float(manuales["monto_max_dia"]))),
        "log_horas_desde_ultima": float(np.log1p(float(ultima_tx["horas_desde_ultima"]))),
    }


def df_a_transacciones(df: pd.DataFrame):
    """Convierte el DataFrame de entrada en list[dict] validando lo básico.

    Lanza EntradaInvalidaError con mensajes claros para el usuario.
    """
    if df is None:
        raise EntradaInvalidaError("No hay transacciones. Agregue al menos una fila.")

    faltantes = [c for c in CAMPOS_TX if c not in df.columns]
    if faltantes:
        raise EntradaInvalidaError(
            f"Faltan columnas en la entrada: {faltantes}. Columnas requeridas: {CAMPOS_TX}."
        )

    df = df[CAMPOS_TX].copy()
    df = df.dropna(how="all")  # filas totalmente vacías del editor
    if df.empty:
        raise EntradaInvalidaError("No hay transacciones. Agregue al menos una fila.")

    for i, (_, fila) in enumerate(df.iterrows(), start=1):
        vacios = [c for c in CAMPOS_TX if pd.isna(fila[c])]
        if vacios:
            raise EntradaInvalidaError(f"La fila {i} tiene campos vacíos: {vacios}.")

    for c in ["monto", "hora", "es_nuevo", "horas_desde_ultima"]:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            raise EntradaInvalidaError(f"La columna '{c}' debe contener solo números.")

    if not df["es_nuevo"].isin([0, 1]).all():
        raise EntradaInvalidaError("La columna 'es_nuevo' solo admite 0 o 1.")

    df["canal"] = df["canal"].astype(str).str.strip()
    df["categoria"] = df["categoria"].astype(str).str.strip()

    if len(df) > MAX_TX:
        st.warning(f"Se recibieron {len(df)} transacciones; se usarán las últimas {MAX_TX}.")
        df = df.tail(MAX_TX)

    txs = df.to_dict("records")
    for t in txs:
        t["monto"] = float(t["monto"])
        t["hora"] = float(t["hora"])
        t["es_nuevo"] = int(t["es_nuevo"])
        t["horas_desde_ultima"] = float(t["horas_desde_ultima"])
    return txs


def leer_csv(archivo) -> pd.DataFrame:
    try:
        contenido = archivo.getvalue().decode("utf-8-sig")
        df = pd.read_csv(io.StringIO(contenido))
    except UnicodeDecodeError:
        raise EntradaInvalidaError("El CSV debe estar codificado en UTF-8.")
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        raise EntradaInvalidaError("No se pudo leer el CSV. Verifique que tenga encabezados y comas.")
    if len(df) > 200:
        raise EntradaInvalidaError("El CSV es demasiado grande (máx. 200 filas); suba solo las últimas transacciones.")
    return df


def mostrar_resultado(res: dict):
    color, fondo, etiqueta = COLORES[res["semaforo"]]
    p = min(max(float(res["puntuacion"]), 0.0), 1.0)
    u = float(res["umbral"])
    st.markdown(
        f"""
        <div style="border:1px solid #ddd6c8;border-left:6px solid {color};border-radius:8px;
                    background:#fffdf9;padding:20px 24px;margin:4px 0 10px 0;">
          <div style="display:flex;flex-wrap:wrap;gap:36px;align-items:flex-end;">
            <div>
              <div style="font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;color:#5b6472;">Puntuación de riesgo</div>
              <div class="num" style="font-size:2.6rem;font-weight:600;color:{color};line-height:1.1;">{res["puntuacion_pct"]}</div>
            </div>
            <div>
              <div style="font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;color:#5b6472;">Decisión</div>
              <div style="font-size:1.6rem;font-weight:700;color:#13233a;line-height:1.3;">{res["decision"].upper()}</div>
            </div>
            <div>
              <div style="font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;color:#5b6472;">Nivel</div>
              <div style="display:inline-block;margin-top:6px;padding:4px 12px;border-radius:4px;
                          background:{fondo};color:{color};font-weight:600;">{etiqueta}</div>
            </div>
            <div>
              <div style="font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;color:#5b6472;">Umbral</div>
              <div class="num" style="font-size:1.3rem;color:#13233a;line-height:1.6;">{u * 100:.1f}%</div>
            </div>
          </div>
          <div style="position:relative;height:8px;background:#ece7dc;border-radius:4px;margin:18px 0 6px 0;">
            <div style="width:{p * 100:.2f}%;height:100%;background:{color};border-radius:4px;"></div>
            <div style="position:absolute;left:{u * 100:.2f}%;top:-5px;width:2px;height:18px;background:#13233a;"></div>
          </div>
          <div style="font-size:.85rem;color:#4a5363;">{res["recomendacion"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Verde si la puntuación es menor a la mitad del umbral, ámbar entre la mitad y el umbral, "
        "rojo si llega al umbral (la transacción se bloquea). La línea oscura en la barra marca el umbral."
    )


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="encabezado">
      <div class="kicker">Banco del Altiplano · Operaciones antifraude</div>
      <div class="titulo">Monitoreo transaccional de fraude</div>
      <div class="sub">Ingrese las últimas transacciones de un cliente (la más reciente al final)
      y sus variables agregadas. El modelo híbrido GRU + Regresión Logística estima la
      probabilidad de fraude de la <b>última</b> transacción.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    AGG_COLS, UMBRAL = cargar_modelo()
except ArtefactosNoEncontradosError as e:
    st.error(f"No se pudieron cargar los artefactos del modelo: {e}")
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Error inesperado al cargar el modelo: {e}")
    st.stop()

EJ = ejemplos()

with st.sidebar:
    st.header("Ejemplos", anchor=False)
    nombre_ej = st.selectbox("Cargar ejemplo", list(EJ.keys()), index=0)
    st.caption(f"Umbral de decisión del modelo: **{UMBRAL:.3f}**")
    st.markdown("---")
    st.caption(
        "Canales: POS, online, ATM.  \nCategorías: comida, retail, viajes, "
        "entretenimiento, servicios.  \nMáx. 10 transacciones."
    )

df_ej, agg_ej = EJ[nombre_ej]
ks = nombre_ej  # el sufijo de las keys reinicia los widgets al cambiar de ejemplo

modo = st.radio("Origen de las transacciones", ["Formulario", "Subir CSV"], horizontal=True)

df_entrada = None
error_entrada = None

if modo == "Formulario":
    st.subheader("1. Últimas transacciones", anchor=False)
    df_entrada = st.data_editor(
        df_ej,
        key=f"editor_{ks}",
        num_rows="dynamic",
        
        column_config={
            "monto": st.column_config.NumberColumn("Monto (Q)", min_value=0.0, format="%.2f"),
            "hora": st.column_config.NumberColumn("Hora (0-23)", min_value=0.0, max_value=23.0),
            "canal": st.column_config.SelectboxColumn("Canal", options=scoring.CHANNELS),
            "categoria": st.column_config.SelectboxColumn("Categoría", options=scoring.MERCHANT_CATS),
            "es_nuevo": st.column_config.NumberColumn("¿Comercio nuevo? (0/1)", min_value=0, max_value=1, step=1),
            "horas_desde_ultima": st.column_config.NumberColumn("Horas desde la anterior", min_value=0.0),
        },
    )
else:
    st.subheader("1. Subir CSV", anchor=False)
    plantilla = df_ej.to_csv(index=False)
    st.download_button("Descargar plantilla CSV", plantilla, "transacciones_ejemplo.csv", "text/csv")
    archivo = st.file_uploader("CSV con columnas: " + ", ".join(CAMPOS_TX), type=["csv"])
    if archivo is not None:
        try:
            df_entrada = leer_csv(archivo)
            st.dataframe(df_entrada)
        except EntradaInvalidaError as e:
            error_entrada = str(e)
            st.error(error_entrada)

st.subheader("2. Variables agregadas del cliente", anchor=False)
cols = st.columns(4)
manuales = {}
for i, (clave, (etiqueta, minimo, _, tipo)) in enumerate(AGG_MANUALES.items()):
    with cols[i % 4]:
        if tipo == "int":
            manuales[clave] = st.number_input(
                etiqueta, min_value=int(minimo), value=int(agg_ej[clave]), step=1, key=f"{clave}_{ks}")
        else:
            manuales[clave] = st.number_input(
                etiqueta, min_value=float(minimo), value=float(agg_ej[clave]), key=f"{clave}_{ks}")
with cols[0]:
    manuales["es_fin_semana"] = int(st.checkbox("Fin de semana", value=bool(agg_ej["es_fin_semana"]), key=f"fds_{ks}"))
with cols[1]:
    manuales["es_comercio_nuevo"] = int(st.checkbox(
        "Comercio nuevo", value=bool(agg_ej["es_comercio_nuevo"]), key=f"cn_{ks}"))
st.caption(
    "La hora, el log del monto y el log de las horas desde la última transacción se "
    "calculan automáticamente a partir de la última fila."
)

st.markdown("---")
if st.button("Calcular riesgo", type="primary"):
    try:
        if error_entrada:
            raise EntradaInvalidaError(error_entrada)
        if df_entrada is None:
            raise EntradaInvalidaError("Suba un CSV o use el formulario antes de calcular.")

        transacciones = df_a_transacciones(df_entrada)
        agregadas = construir_agregadas(transacciones[-1], manuales)
        resultado = evaluar_riesgo(transacciones, agregadas)

        st.subheader("Resultado", anchor=False)
        mostrar_resultado(resultado)
        registrar_uso("prediccion", origen=modo, ejemplo=nombre_ej, n_tx=len(transacciones),
                      puntuacion=round(float(resultado["puntuacion"]), 6),
                      decision=resultado["decision"], semaforo=resultado["semaforo"],
                      umbral=resultado["umbral"])

        with st.expander("Ver datos enviados al modelo"):
            st.json({"transacciones": transacciones, "agregadas": agregadas})
    except EntradaInvalidaError as e:
        st.error(f"Entrada inválida: {e}")
        registrar_uso("entrada_invalida", origen=modo, ejemplo=nombre_ej, detalle=str(e))
    except ArtefactosNoEncontradosError as e:
        st.error(f"Faltan archivos del modelo: {e}")
    except Exception as e:  # noqa: BLE001
        st.error(f"Ocurrió un error inesperado al calcular el riesgo: {e}")
