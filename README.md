---
title: Monitoreo de Fraude Transaccional
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
---

# Monitoreo transaccional de fraude

App Streamlit que puntúa el riesgo de fraude de la última transacción de un cliente
usando un modelo híbrido GRU (NumPy) + Regresión Logística.

- Entrada: últimas transacciones (formulario o CSV) + variables agregadas.
- Salida: puntuación, semáforo (verde / ámbar / rojo) y decisión (autorizar / bloquear).
- Umbral de decisión: 0.075 (costo FN Q4,200 vs FP Q180).

Proyecto Final — Deep Learning, Universidad del Valle de Guatemala.
