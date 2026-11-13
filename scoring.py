"""
scoring.py — Inferencia del modelo C (híbrido GRU + Regresión Logística)
para detección de fraude transaccional.

Extraído del notebook Proyecto 1 (modelo C validado).
SOLO INFERENCIA: no reentrena nada. Carga los artefactos ya entrenados.

Entrada
-------
  - transacciones : lista de dict (o filas ordenadas cronológicamente, más
    reciente al final) con las últimas N (<= 10) transacciones del cliente.
    Cada una requiere: monto, hora, canal, categoria, es_nuevo,
    horas_desde_ultima.
  - agregadas : dict/list con las 13 variables agregadas del cliente en el
    orden de artefactos/columnas_agregadas.json.

Salida
------
  dict con: puntuacion (float en [0,1]), decision ("bloquear"/"autorizar"),
  umbral usado.
"""

import json
import os
import pickle
from typing import Any, Dict, List, Union

import numpy as np

# ---------------------------------------------------------------------------
# Rutas de artefactos (relativas a este archivo, para que funcione desde
# cualquier directorio de trabajo).
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artefactos")

# ---------------------------------------------------------------------------
# Constantes del modelo (coherentes con el notebook Proyecto 1).
# ---------------------------------------------------------------------------
W_COMPLETO = 10  # ventana de historia del GRU

CHANNELS = ["POS", "online", "ATM"]
MERCHANT_CATS = ["comida", "retail", "viajes", "entretenimiento", "servicios"]

# Las 13 columnas agregadas en el orden exacto del scaler (del JSON).
AGG_COLS: List[str] = []

# Cargado perezoso al importar.
_GRU = None
_MODEL_C = None
_SCALER_AGG = None
_UMBRAL = 0.075


# ---------------------------------------------------------------------------
# Utilidades numéricas (idénticas al notebook).
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Carga de artefactos.
# ---------------------------------------------------------------------------
def _cargar_artefactos() -> None:
    """Carga los 5 artefactos desde ART_DIR y deja los objetos listos."""
    global _GRU, _MODEL_C, _SCALER_AGG, _UMBRAL, AGG_COLS

    # 1) Pesos del GRU (modelo B) — forward-only.
    npz_path = os.path.join(ART_DIR, "modelo_B_gru_pesos.npz")
    with np.load(npz_path) as gru_npz:
        pesos = {k: gru_npz[k] for k in gru_npz.files
                 if k not in ("hidden_dim", "input_dim")}
        hidden_dim = int(gru_npz["hidden_dim"])
        input_dim = int(gru_npz["input_dim"])

    _GRU = _GRUForwardOnly(pesos, hidden_dim=hidden_dim, input_dim=input_dim)

    # 2) Modelo C — Regresión Logística.
    with open(os.path.join(ART_DIR, "modelo_C_logreg.pkl"), "rb") as f:
        _MODEL_C = pickle.load(f)

    # 3) Scaler de agregadas (ajustado solo con train).
    with open(os.path.join(ART_DIR, "scaler_agg.pkl"), "rb") as f:
        _SCALER_AGG = pickle.load(f)

    # 4) Columnas agregadas (orden fijo del scaler).
    with open(os.path.join(ART_DIR, "columnas_agregadas.json"),
              "r", encoding="utf-8") as f:
        AGG_COLS = json.load(f)

    # 5) Parámetros (umbral de decisión, costos, etc.).
    with open(os.path.join(ART_DIR, "parametros_preparacion.json"),
              "r", encoding="utf-8") as f:
        parametros = json.load(f)
    _UMBRAL = float(parametros.get("umbral_decision", 0.075))


class _GRUForwardOnly:
    """Forward-only del GRU binario (sin backprop).
    Replica la arquitectura del notebook: input_dim=14, hidden_dim=24."""

    def __init__(self, pesos: dict, hidden_dim: int, input_dim: int):
        self.Wz = pesos["Wz"]; self.Uz = pesos["Uz"]; self.bz = pesos["bz"]
        self.Wr = pesos["Wr"]; self.Ur = pesos["Ur"]; self.br = pesos["br"]
        self.Wh = pesos["Wh"]; self.Uh = pesos["Uh"]; self.bh = pesos["bh"]
        self.Wo = pesos["Wo"]; self.bo = pesos["bo"]
        self.H = hidden_dim
        self.F = input_dim

    def forward(self, X: np.ndarray):
        """X: (B, T, F). Devuelve (proba, hidden_final)."""
        B, T, F = X.shape
        h = np.zeros((B, self.H), dtype=np.float64)
        for t in range(T):
            xt = X[:, t, :]
            z = sigmoid(xt @ self.Wz + h @ self.Uz + self.bz)
            r = sigmoid(xt @ self.Wr + h @ self.Ur + self.br)
            hh = np.tanh(xt @ self.Wh + (r * h) @ self.Uh + self.bh)
            h = (1.0 - z) * h + z * hh
        logits = (h @ self.Wo + self.bo).ravel()
        return sigmoid(logits), h


# ---------------------------------------------------------------------------
# Construcción de entrada del modelo.
# ---------------------------------------------------------------------------
def _vector_paso(monto, hora, canal, categoria, es_nuevo, horas_desde_ultima):
    """Vector de 14 dims para un paso (idéntico al notebook).

    v[0]   = log1p(monto)
    v[1:3] = sin/cos(hora) codificación cíclica
    v[3]   = log1p(horas_desde_ultima)
    v[4]   = es_nuevo (0/1)
    v[5:8] = one-hot canal (POS, online, ATM)
    v[8:13]= one-hot categoría (comida, retail, viajes, entretenimiento, servicios)
    """
    v = np.zeros(14, dtype=np.float32)
    v[0] = np.log1p(float(monto))
    v[1] = np.sin(2.0 * np.pi * float(hora) / 24.0)
    v[2] = np.cos(2.0 * np.pi * float(hora) / 24.0)
    v[3] = np.log1p(float(horas_desde_ultima))
    v[4] = float(es_nuevo)
    v[5 + CHANNELS.index(canal)] = 1.0
    v[8 + MERCHANT_CATS.index(categoria)] = 1.0
    return v


def _construir_secuencia(transacciones: List[dict], W: int = W_COMPLETO) -> np.ndarray:
    """Construye (1, W, 14) con relleno a la izquierda (sin fuga de futuro).

    transacciones : lista ordenada cronológicamente, la más reciente al final.
    Si hay menos de W pasos, los primeros se rellenan con ceros.
    """
    pasos = [_vector_paso(
        monto=tx["monto"],
        hora=tx["hora"],
        canal=tx["canal"],
        categoria=tx["categoria"],
        es_nuevo=tx["es_nuevo"],
        horas_desde_ultima=tx["horas_desde_ultima"],
    ) for tx in transacciones]

    hist = pasos[-W:]  # sólo los últimos W pasos
    seq = np.zeros((W, 14), dtype=np.float32)
    seq[W - len(hist):] = np.asarray(hist, dtype=np.float32)
    return seq[np.newaxis, :, :]  # (1, W, 14)


def _extraer_agregadas(agregadas: Union[dict, list, np.ndarray]) -> np.ndarray:
    """Devuelve las 13 agregadas en el orden del scaler, como (1, 13)."""
    if isinstance(agregadas, dict):
        try:
            valores = [agregadas[col] for col in AGG_COLS]
        except KeyError as e:
            raise KeyError(
                f"Falta la columna agregada '{e.args[0]}'. "
                f"Columnas esperadas ({len(AGG_COLS)}): {AGG_COLS}"
            )
    else:
        valores = list(agregadas)

    if len(valores) != len(AGG_COLS):
        raise ValueError(
            f"agregadas debe tener {len(AGG_COLS)} valores, se recibieron {len(valores)}. "
            f"Orden esperado: {AGG_COLS}"
        )
    return np.asarray(valores, dtype=np.float64).reshape(1, -1)


# ---------------------------------------------------------------------------
# API principal.
# ---------------------------------------------------------------------------
def score_transaccion(
    transacciones: List[dict],
    agregadas: Union[dict, list, np.ndarray],
    umbral: float = None,
) -> dict:
    """Calcula puntuación de riesgo y decisión para una transacción.

    Parámetros
    ----------
    transacciones : list[dict]
        Últimas N (<= 10) transacciones del cliente, ordenadas cronológicamente.
        Cada dict: monto, hora, canal, categoria, es_nuevo, horas_desde_ultima.
    agregadas : dict | list | np.ndarray
        13 variables agregadas del cliente (orden de columnas_agregadas.json).
    umbral : float, opcional
        Umbral de decisión. Por defecto usa el de parametros_preparacion.json (0.075).

    Retorna
    -------
    dict con claves:
        puntuacion : float en [0, 1]
        decision   : "bloquear" | "autorizar"
        umbral     : float usado
    """
    # Asegurar que los artefactos estén cargados.
    if _GRU is None:
        _cargar_artefactos()

    if umbral is None:
        umbral = _UMBRAL

    # --- 1) Secuencia del GRU -> estado oculto final (24) ---
    X_seq = _construir_secuencia(transacciones, W=W_COMPLETO)
    _, hidden = _GRU.forward(X_seq)  # hidden: (1, 24)

    # --- 2) Agregadas escaladas (13) ---
    Xagg = _extraer_agregadas(agregadas)
    agg_scaled = _SCALER_AGG.transform(Xagg)  # (1, 13)

    # --- 3) Concatena [hidden_state, agg_scaled] -> 37 features -> predict_proba ---
    Xc = np.hstack([hidden, agg_scaled])  # (1, 37)
    puntuacion = float(_MODEL_C.predict_proba(Xc)[:, 1][0])

    decision = "bloquear" if puntuacion >= umbral else "autorizar"

    return {
        "puntuacion": round(puntuacion, 6),
        "decision": decision,
        "umbral": umbral,
    }


def score_desde_ejemplo(monto, hora, canal, categoria, es_nuevo,
                        horas_desde_ultima, agregadas, umbral=None):
    """Conveniencia: puntúa una única transacción con sus agregadas.
    Construye internamente una secuencia de 1 paso (relleno a la izquierda)."""
    tx = {
        "monto": monto,
        "hora": hora,
        "canal": canal,
        "categoria": categoria,
        "es_nuevo": es_nuevo,
        "horas_desde_ultima": horas_desde_ultima,
    }
    return score_transaccion([tx], agregadas, umbral=umbral)


# ---------------------------------------------------------------------------
# Ejemplo rápido de uso (solo se ejecuta si se corre el archivo directamente).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo sintético de una transacción legítima (bajo riesgo).
    tx_legitima = {
        "monto": 120.0,
        "hora": 14,
        "canal": "POS",
        "categoria": "comida",
        "es_nuevo": 0,
        "horas_desde_ultima": 24.0,
    }
    agregadas_legitima = {
        "n_tx_1h": 1,
        "n_tx_24h": 3,
        "div_comercio_7d": 2,
        "div_canal_7d": 1,
        "ratio_monto_hist": 1.0,
        "n_tx_historicas": 50,
        "hora": 14,
        "es_fin_semana": 0,
        "es_comercio_nuevo": 0,
        "log_monto": np.log1p(120.0),
        "log_monto_prom_24h": np.log1p(100.0),
        "log_monto_max_dia": np.log1p(150.0),
        "log_horas_desde_ultima": np.log1p(24.0),
    }

    # Ejemplo sintético de una transacción sospechosa (alto riesgo).
    tx_fraude = {
        "monto": 5000.0,
        "hora": 3,
        "canal": "ATM",
        "categoria": "viajes",
        "es_nuevo": 1,
        "horas_desde_ultima": 0.2,
    }
    agregadas_fraude = {
        "n_tx_1h": 5,
        "n_tx_24h": 8,
        "div_comercio_7d": 6,
        "div_canal_7d": 3,
        "ratio_monto_hist": 50.0,
        "n_tx_historicas": 20,
        "hora": 3,
        "es_fin_semana": 1,
        "es_comercio_nuevo": 1,
        "log_monto": np.log1p(5000.0),
        "log_monto_prom_24h": np.log1p(200.0),
        "log_monto_max_dia": np.log1p(800.0),
        "log_horas_desde_ultima": np.log1p(0.2),
    }

    print("=== scoring.py — ejemplo de uso ===")
    print(f"Umbral de decisión: {_UMBRAL}")
    print()

    r1 = score_transaccion([tx_legitima], agregadas_legitima)
    print(f"Legítima  : puntuación={r1['puntuacion']} -> {r1['decision']}")

    r2 = score_transaccion([tx_fraude], agregadas_fraude)
    print(f"Sospechosa: puntuación={r2['puntuacion']} -> {r2['decision']}")
