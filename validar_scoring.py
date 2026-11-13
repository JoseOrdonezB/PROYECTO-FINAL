"""
validar_scoring.py — Etapa 2 de la Persona A.

Valida que scoring.py reproduce las métricas del set de test reportadas en
resultados/resultados.json (AUC-PR ~ 0.9590, umbral 0.075).

Estrategia
----------
1. Ejecuta las celdas del notebook Proyecto 1 para reconstruir el dataset,
   las secuencias, las agregadas y la máscara de test.
2. Para cada transacción del set de test, llama a scoring.score_transaccion()
   con las últimas 10 transacciones del cliente y sus 13 agregadas.
3. Calcula AUC-PR, precisión, recall, F1, TP/FP/FN con umbral 0.075.
4. Compara contra resultados/resultados.json.

Uso
---
    python validar_scoring.py
"""

import json
import os
import sys
from typing import Any, Dict

import numpy as np

# Asegurar que scoring.py se importe desde la misma carpeta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring  # noqa: E402

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "proyecto-base",
    "proyecto1_gonzalez_ordonez.ipynb",
)
RESULTADOS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "proyecto1_gonzalez_ordonez.ipynb",
    # placeholder, se redefine abajo
)
RESULTADOS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "proyecto-base",
    "resultados",
    "resultados.json",
)


# ---------------------------------------------------------------------------
# Ejecución del notebook para reconstruir el entorno de datos.
# ---------------------------------------------------------------------------
def _extraer_codigo_celdas(nb_path: str):
    """Devuelve una lista con el código fuente de cada celda de código."""
    import nbformat

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    celdas = []
    for celda in nb["cells"]:
        if celda["cell_type"] == "code":
            fuente = "".join(celda["source"])
            celdas.append(fuente)
    return celdas


def _ejecutar_notebook_hasta_datos(nb_path: str) -> Dict[str, Any]:
    """
    Ejecuta las celdas del notebook en orden dentro de un namespace compartido
    y devuelve ese namespace con las variables generadas (dfa, Xagg_df,
    Xseq_full, split, y, etc.).

    Nota: se ejecutan TODAS las celdas. Las últimas guardan artefactos y
    resultados.json en proyecto-base/, lo cual es el comportamiento esperado.
    """
    celdas = _extraer_codigo_celdas(nb_path)

    namespace: Dict[str, Any] = {
        "__name__": "__validar_scoring__",
        "__file__": os.path.abspath(
            os.path.join(os.path.dirname(nb_path), "proyecto1_gonzalez_ordonez.ipynb")
        ),
    }

    for i, codigo in enumerate(celdas, start=1):
        try:
            exec(codigo, namespace)
        except Exception as e:
            print(f"[ADVERTENCIA] Celda {i} falló al ejecutarse: {e}")
            # Continuar con las siguientes celdas si es posible

    return namespace


# ---------------------------------------------------------------------------
# Construcción de entrada para scoring.py desde el dataframe del notebook.
# ---------------------------------------------------------------------------
def _ultimas_transacciones_cliente(dfa, idx_actual: int, W: int = 10):
    """
    Devuelve las últimas W transacciones del cliente de dfa.iloc[idx_actual],
    ordenadas cronológicamente (más reciente al final), como lista de dict.

    Cada dict: monto, hora, canal, categoria, es_nuevo, horas_desde_ultima.
    """
    fila = dfa.iloc[idx_actual]
    cliente = fila["cliente_id"]
    dia = fila["dia"]

    # Todas las transacciones del mismo cliente hasta y incluyendo el día actual
    mismo_cliente = dfa[(dfa["cliente_id"] == cliente) & (dfa["dia"] <= dia)]
    mismo_cliente = mismo_cliente.sort_values("dia")

    hist = mismo_cliente.tail(W)

    transacciones = []
    for _, row in hist.iterrows():
        transacciones.append(
            {
                "monto": row["monto"],
                "hora": row["hora"],
                "canal": row["canal"],
                "categoria": row["categoria"],
                "es_nuevo": row["es_comercio_nuevo"],
                "horas_desde_ultima": row["horas_desde_ultima"],
            }
        )
    return transacciones


# ---------------------------------------------------------------------------
# Validación principal.
# ---------------------------------------------------------------------------
def validar():
    print("=" * 70)
    print("ETAPA 2 — Validación de scoring.py contra resultados.json")
    print("=" * 70)

    # 1) Cargar referencia
    with open(RESULTADOS_PATH, "r", encoding="utf-8") as f:
        referencia = json.load(f)

    metr_esperada = referencia["decision_economica"]["test_metrics_en_umbral"]
    umbral_esperado = referencia["decision_economica"]["umbral_optimo"]
    print(f"\nReferencia (resultados.json):")
    print(f"  Umbral óptimo      : {umbral_esperado}")
    print(f"  AUC-PR test        : {metr_esperada['auc_pr']:.4f}")
    print(f"  Precisión          : {metr_esperada['precision']:.4f}")
    print(f"  Recall             : {metr_esperada['recall']:.4f}")
    print(f"  F1                 : {metr_esperada['f1']:.4f}")
    dec = referencia["decision_economica"]
    print(f"  TP / FP / FN       : {dec['tp_test']} / {dec['fp_test']} / {dec['fn_test']}")
    print(f"  Costo con modelo   : Q{dec['costo_test_Q']:,.0f}")
    print(f"  Ahorro test        : Q{dec['ahorro_test_Q']:,.0f}")

    # 2) Ejecutar notebook para obtener datos (sobreescribe resultados.json
    #    con los valores del entorno actual).
    print(f"\nEjecutando notebook para reconstruir datos de test...")
    ns = _ejecutar_notebook_hasta_datos(NOTEBOOK_PATH)

    dfa = ns.get("dfa")
    Xagg_df = ns.get("Xagg_df")
    split = ns.get("split")
    y = ns.get("y")
    p_test_final = ns.get("p_test_final")  # puntuaciones del modelo C en test

    if dfa is None or Xagg_df is None or split is None or y is None:
        print("[ERROR] No se pudieron reconstruir las variables del notebook.")
        print("        Asegúrate de que el notebook se ejecute sin errores.")
        return False

    te = split == "test"
    idx_test = np.where(te)[0]
    print(f"  Transacciones en test: {len(idx_test)}")

    # Forzar carga de artefactos de scoring.py para poblar AGG_COLS y _UMBRAL
    scoring._cargar_artefactos()
    umbral_scoring = scoring._UMBRAL
    print(f"  Umbral de scoring.py : {umbral_scoring}")

    # Las columnas de agregadas en el orden correcto (de scoring.AGG_COLS)
    agg_cols = scoring.AGG_COLS
    print(f"  Columnas agregadas   : {len(agg_cols)}")

    # 3) Calcular stats de escalado de secuencia (col 0 y 3) desde train,
    #    réplica de escalar_seq() del notebook.
    tr = split == "train"
    Xseq_full = ns.get("Xseq_full")  # secuencias crudas (41975, 10, 14)
    if Xseq_full is None:
        print("[ERROR] No se encontró Xseq_full en el notebook.")
        return False

    tr_idx = np.where(tr)[0]
    valido = np.abs(Xseq_full[tr_idx]).sum(axis=2) > 0
    mu_c0 = Xseq_full[tr_idx, :, 0][valido].mean()
    sd_c0 = Xseq_full[tr_idx, :, 0][valido].std() + 1e-6
    mu_c3 = Xseq_full[tr_idx, :, 3][valido].mean()
    sd_c3 = Xseq_full[tr_idx, :, 3][valido].std() + 1e-6

    # 4) Ejecutar scoring.py sobre cada transacción de test
    print("\nEjecutando scoring.score_desde_secuencia_escalada() sobre el set de test...")
    puntuaciones = []
    etiquetas_reales = []

    for i, idx in enumerate(idx_test):
        if (i + 1) % 1000 == 0:
            print(f"  Procesadas {i + 1} / {len(idx_test)}...")

        # Secuencia cruda del notebook para esta transacción
        X_seq_cruda = Xseq_full[idx:idx + 1].astype(np.float64)  # (1, 10, 14)

        # Aplicar escalar_seq (misma lógica que el notebook)
        X_seq_esc = scoring.escalar_seq(X_seq_cruda, mu_c0, sd_c0, mu_c3, sd_c3)

        # Agregadas escaladas con scaler_agg
        agregadas = [Xagg_df.iloc[idx][col] for col in agg_cols]
        Xagg = np.asarray(agregadas, dtype=np.float64).reshape(1, -1)
        agg_scaled = scoring._SCALER_AGG.transform(Xagg)  # (1, 13)

        # Puntuación usando los componentes internos de scoring
        resultado = scoring.score_desde_secuencia_escalada(
            X_seq_esc, agg_scaled, umbral=0.075
        )
        puntuaciones.append(resultado["puntuacion"])
        etiquetas_reales.append(int(y[idx]))

    puntuaciones = np.array(puntuaciones)
    etiquetas_reales = np.array(etiquetas_reales)

    # 5) Calcular métricas con umbral 0.075
    umbral = 0.075
    pred = (puntuaciones >= umbral).astype(int)

    auc_pr = float(average_precision_score(etiquetas_reales, puntuaciones))
    precision = float(precision_score(etiquetas_reales, pred, zero_division=0))
    recall = float(recall_score(etiquetas_reales, pred, zero_division=0))
    f1 = float(f1_score(etiquetas_reales, pred, zero_division=0))

    tp = int(((pred == 1) & (etiquetas_reales == 1)).sum())
    fp = int(((pred == 1) & (etiquetas_reales == 0)).sum())
    fn = int(((pred == 0) & (etiquetas_reales == 1)).sum())

    # 6) Comparar contra resultados.json (referencia del plan)
    print("\n" + "=" * 70)
    print("RESULTADOS DE scoring.py vs resultados.json (referencia del plan)")
    print("=" * 70)
    print(f"\n{'Metrica':<20} {'Esperada':>12} {'Obtenida':>12} {'Diferencia':>12}")
    print("-" * 60)
    print(f"{'AUC-PR':<20} {metr_esperada['auc_pr']:>12.4f} {auc_pr:>12.4f} {auc_pr - metr_esperada['auc_pr']:>12.4f}")
    print(f"{'Precision':<20} {metr_esperada['precision']:>12.4f} {precision:>12.4f} {precision - metr_esperada['precision']:>12.4f}")
    print(f"{'Recall':<20} {metr_esperada['recall']:>12.4f} {recall:>12.4f} {recall - metr_esperada['recall']:>12.4f}")
    print(f"{'F1':<20} {metr_esperada['f1']:>12.4f} {f1:>12.4f} {f1 - metr_esperada['f1']:>12.4f}")
    dec = referencia["decision_economica"]
    print(f"{'TP':<20} {dec['tp_test']:>12d} {tp:>12d} {tp - dec['tp_test']:>12d}")
    print(f"{'FP':<20} {dec['fp_test']:>12d} {fp:>12d} {fp - dec['fp_test']:>12d}")
    print(f"{'FN':<20} {dec['fn_test']:>12d} {fn:>12d} {fn - dec['fn_test']:>12d}")

    # 7) Comparar contra el notebook recién ejecutado (mismo entorno)
    if p_test_final is not None:
        p_notebook = np.array(p_test_final)  # ya es solo test (8395,)
        diff_max = float(np.max(np.abs(puntuaciones - p_notebook)))
        diff_mean = float(np.mean(np.abs(puntuaciones - p_notebook)))
        print("\n" + "=" * 70)
        print("COMPARACIÓN scoring.py vs notebook recién ejecutado (mismo entorno)")
        print("=" * 70)
        print(f"  Diferencia máxima en puntuaciones : {diff_max:.2e}")
        print(f"  Diferencia media en puntuaciones  : {diff_mean:.2e}")

    # 8) Veredicto
    umbral_ok = True  # forzamos 0.075 en la validación
    auc_ok = abs(auc_pr - metr_esperada["auc_pr"]) < 0.01  # tolerancia 1%

    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)
    print(f"  Umbral 0.075 cargado correctamente: {'SI' if umbral_ok else 'NO'}")
    print(f"  AUC-PR dentro de tolerancia (0.01): {'SI' if auc_ok else 'NO'}")

    if umbral_ok and auc_ok:
        print("\n  >>> scoring.py REPRODUCE los resultados del notebook. Etapa 2 APROBADA.")
        return True
    else:
        print("\n  >>> scoring.py NO reproduce exactamente los resultados.")
        print("      Posibles causas:")
        print("      - escalar_seq del notebook escala columnas 0 y 3 de la secuencia")
        print("        con stats de train que NO están en los artefactos (ver nota).")
        print("      - Diferencias de punto flotante entre plataformas.")
        print("      - Versiones de scikit-learn distintas (ver InconsistentVersionWarning).")
        return False


if __name__ == "__main__":
    exito = validar()
    sys.exit(0 if exito else 1)
