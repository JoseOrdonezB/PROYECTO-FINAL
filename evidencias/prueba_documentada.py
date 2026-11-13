"""
Prueba documentada: entrada -> salida del modelo C usando scoring.py directamente.

Usa los mismos ejemplos que trae precargados la app y además tres entradas
inválidas para mostrar que el contrato rechaza datos malos con un mensaje claro.

Uso (desde la raíz del repo):
    python evidencias/prueba_documentada.py
Genera evidencias/prueba_documentada.json
"""
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import numpy as np  # noqa: E402
from scoring import EntradaInvalidaError, evaluar_riesgo  # noqa: E402

# Mismos ejemplos precargados que app.py (copiados aquí para no importar Streamlit).
CAMPOS = ["canal", "categoria", "monto", "hora", "es_nuevo", "horas_desde_ultima"]
HIST = [
    ("POS", "comida", 85, 14, 0, 60), ("POS", "retail", 110, 16, 0, 72),
    ("POS", "comida", 70, 13, 0, 55), ("POS", "comida", 95, 15, 0, 80),
    ("online", "retail", 120, 17, 0, 66), ("POS", "comida", 80, 14, 0, 90),
    ("POS", "servicios", 100, 15, 0, 48), ("POS", "comida", 90, 13, 0, 70),
    ("POS", "retail", 105, 16, 0, 62),
]


def _txs(filas):
    return [{k: (float(v) if k in ("monto", "hora", "horas_desde_ultima") else v)
             for k, v in zip(CAMPOS, f)} for f in filas]


def _agg(ultima, **kw):
    """Igual que construir_agregadas() de app.py."""
    m = dict(n_tx_1h=1, n_tx_24h=1, div_comercio_7d=2, div_canal_7d=1, ratio_monto_hist=1.0,
             n_tx_historicas=24, es_fin_semana=0, es_comercio_nuevo=0,
             monto_prom_24h=88.0, monto_max_dia=88.0)
    m.update(kw)
    return {
        "n_tx_1h": m["n_tx_1h"], "n_tx_24h": m["n_tx_24h"],
        "div_comercio_7d": m["div_comercio_7d"], "div_canal_7d": m["div_canal_7d"],
        "ratio_monto_hist": m["ratio_monto_hist"], "n_tx_historicas": m["n_tx_historicas"],
        "hora": float(ultima["hora"]), "es_fin_semana": m["es_fin_semana"],
        "es_comercio_nuevo": m["es_comercio_nuevo"],
        "log_monto": float(np.log1p(ultima["monto"])),
        "log_monto_prom_24h": float(np.log1p(m["monto_prom_24h"])),
        "log_monto_max_dia": float(np.log1p(m["monto_max_dia"])),
        "log_horas_desde_ultima": float(np.log1p(ultima["horas_desde_ultima"])),
    }


def ejemplos_scoring():
    legit = _txs(HIST + [("POS", "comida", 88, 14, 0, 75)])
    fraude = _txs(HIST[-5:] + [
        ("online", "retail", 12, 3, 1, 60), ("online", "entretenimiento", 25, 3, 1, 0.05),
        ("online", "servicios", 30, 3, 1, 0.07), ("online", "viajes", 18, 3, 1, 0.05),
        ("online", "retail", 1400, 3, 1, 0.1)])
    limite = _txs(HIST + [("POS", "comida", 95, 14, 0, 60)])
    return {
        "Transacción legítima": (legit, _agg(legit[-1], ratio_monto_hist=0.98)),
        "Transacción fraudulenta (escalada rápida)": (fraude, _agg(
            fraude[-1], n_tx_1h=5, n_tx_24h=5, div_comercio_7d=4, div_canal_7d=2,
            ratio_monto_hist=15.0, es_comercio_nuevo=1, monto_prom_24h=297.0,
            monto_max_dia=1400.0)),
        "Caso límite (zona ámbar)": (limite, _agg(limite[-1], monto_prom_24h=95.0,
                                                  monto_max_dia=95.0)),
    }

casos = []

for nombre, (txs, agg) in ejemplos_scoring().items():
    res = evaluar_riesgo(txs, agg)
    casos.append({"caso": nombre, "entrada": {"transacciones": txs, "agregadas": agg},
                  "salida": {k: res[k] for k in ("puntuacion", "puntuacion_pct", "decision",
                                                 "semaforo", "umbral")}})

txs_ok, agg_ok = ejemplos_scoring()["Transacción legítima"]
invalidos = {
    "Error: transacción sin el campo 'monto'": (
        [{k: v for k, v in txs_ok[-1].items() if k != "monto"}], agg_ok),
    "Error: canal desconocido ('WEB')": (
        txs_ok[:-1] + [{**txs_ok[-1], "canal": "WEB"}], agg_ok),
    "Error: lista de transacciones vacía": ([], agg_ok),
}
for nombre, (txs, agg) in invalidos.items():
    try:
        evaluar_riesgo(txs, agg)
        salida = {"error": None}
    except EntradaInvalidaError as e:
        salida = {"error": type(e).__name__, "mensaje": str(e)}
    casos.append({"caso": nombre, "entrada": {"transacciones": txs, "agregadas": agg},
                  "salida": salida})

destino = os.path.join(RAIZ, "evidencias", "prueba_documentada.json")
with open(destino, "w", encoding="utf-8") as f:
    json.dump(casos, f, ensure_ascii=False, indent=2)

for c in casos:
    s = c["salida"]
    if "decision" in s:
        print(f"{c['caso']:<45} -> {s['puntuacion_pct']:>7}  {s['semaforo']:<6} {s['decision'].upper()}")
    else:
        print(f"{c['caso']:<45} -> {s.get('error')}: {s.get('mensaje')}")
print(f"\nGuardado en {destino}")
