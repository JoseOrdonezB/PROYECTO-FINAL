---
title: Monitoreo Transaccional de Fraude
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
short_description: Modelo hibrido GRU + Regresion Logistica para fraude
---

# Monitoreo transaccional de fraude

App del Proyecto Final de Deep Learning (UVG, 2026). Estima la probabilidad de fraude de la
última transacción de un cliente con el modelo C del Proyecto 1 (estado oculto de una GRU +
variables agregadas → regresión logística) y decide si autorizar o bloquear (umbral 0.075).

> Nota: este Space no se usó al final. Al crearlo, Hugging Face respondió `402 Payment
> Required` (los Spaces Docker/Gradio en CPU gratis ahora piden PRO). La app está desplegada
> en Streamlit Community Cloud; ver el README principal.

Código y documentación: https://github.com/JoseOrdonezB/PROYECTO-FINAL
