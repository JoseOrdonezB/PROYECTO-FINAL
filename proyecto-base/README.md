# Proyecto 1 — Monitoreo transaccional: detectar lo que el orden revela

Deep Learning, Universidad del Valle, 2026. José Ordoñez (231329) · Adrián González (23152).

## Reproducción

Requisitos: Python 3.12, y las siguientes versiones (las que se usaron para generar
todos los resultados de esta entrega):

```
numpy==2.4.4
pandas==3.0.2
scikit-learn==1.8.0
matplotlib==3.10.8
nbformat==5.11.1
jupyter_client, nbconvert, ipykernel  (para re-ejecutar el notebook)
```

Pasos:

```bash
pip install numpy pandas scikit-learn matplotlib nbformat jupyter_client nbconvert ipykernel

# Re-ejecutar el notebook de arriba a abajo: genera los datos (Ruta A),
# entrena los modelos A/B/C, corre las dos pruebas de falsificacion, verifica el
# gradiente del GRU, calcula el umbral y el analisis economico, y guarda
# resultados/resultados.json, figuras/*.png y artefactos/*
jupyter nbconvert --to notebook --execute --inplace proyecto1_gonzalez_ordonez.ipynb
```

Todo es determinístico: la semilla fija (`SEED = 42`, definida en la primera celda del
notebook) produce siempre el mismo dataset, las mismas particiones y (salvo diferencias
de punto flotante entre plataformas) los mismos resultados. El chequeo de backpropagation
del GRU se hace dentro del notebook mediante gradiente numérico (error relativo máximo
≈ 1.7e-8) y su `assert` detiene la ejecución si el gradiente analítico no coincide.

Todo el código (generador, variables, GRU, entrenamiento, pruebas y guardado) está
consolidado dentro del notebook, que es la única fuente de verdad.

### Estructura del repositorio

```
proyecto1_gonzalez_ordonez.ipynb  - notebook ejecutado con A, B, la apuesta C, las dos
                                     pruebas de falsificacion y el analisis economico
                                     (fuente de verdad; al ejecutarse regenera figuras,
                                     artefactos y resultados)
figuras/                           - 5 figuras PNG generadas por el notebook
artefactos/                        - modelo A (HGB), pesos del GRU (B), modelo C
                                     (logreg), scaler, columnas y parametros
resultados/resultados.json         - todas las metricas numericas citadas en el informe
informe.pdf                        - informe para el comite de riesgos (max. 7 paginas)
presentacion.pdf                   - presentacion (8 diapositivas)
guion_presentacion.md              - guion y notas del presentador para la exposicion
```

## Declaración de uso de IA

Se usó Claude (Anthropic) como asistente de programación para todo el desarrollo:
generación de datos sintéticos, implementación del GRU en NumPy, el pipeline dentro del
notebook, la generación del informe y de la presentación. Todas las salidas numéricas se
tomaron de la ejecución real del notebook y se verificaron contra
`resultados/resultados.json` antes de escribirse en el informe y la presentación —
ninguna cifra fue inventada; todas provienen de la corrida con la semilla fija indicada
arriba.

Tres decisiones técnicas importantes, alternativas consideradas y evidencia que inclinó la decisión:

1. **Implementar el GRU en NumPy en vez de usar PyTorch/TensorFlow.**
   Alternativa considerada: instalar `torch` vía pip. Evidencia: el paquete pesa
   ~530 MB (más las dependencias de CUDA de PyPI, potencialmente varios GB) y el
   entorno de ejecución solo tiene salida de red hacia PyPI/npm/GitHub, sin acceso a
   `download.pytorch.org`; instalarlo no era práctico. Se implementó el GRU a mano y
   se verificó su backpropagation con un chequeo de gradiente numérico (dentro del
   notebook, error relativo máximo ≈ 1.7e-8) antes de usarlo para cualquier resultado
   reportado.

2. **Etiquetar solo la transacción "de impacto" en cada episodio de fraude, no todo el episodio.**
   Alternativa considerada: marcar como fraude todas las transacciones del episodio
   (por ejemplo, las compras pequeñas de prueba en `escalada_rapida`). Evidencia:
   el enunciado pide predecir, transacción por transacción, si *esa* transacción es
   la que debe bloquearse; etiquetar todo el episodio habría inflado artificialmente
   el número de positivos y mezclado transacciones de prueba (pequeñas, ambiguas)
   con la transacción que realmente causa la pérdida.

3. **Definir la prueba de permutación barajando solo la historia, dejando fija la
   transacción evaluada.**
   Alternativa considerada: barajar la secuencia completa, incluyendo qué transacción
   queda en la última posición. Evidencia: el modelo B usa el último paso de la
   secuencia como "la transacción actual" para decidir; barajar esa posición cambiaría
   qué transacción se está clasificando, confundiendo dos preguntas distintas (¿importa
   el orden de la historia? vs. ¿importa cuál transacción se evalúa?). Fijar el último
   paso aísla la pregunta que pide el enunciado.

## Candidato al Proyecto Final

- **Modelo que conservaría:** el candidato C (híbrido: estado oculto del GRU secuencial
  concatenado con las variables agregadas del cliente, alimentando una regresión
  logística). Fue el de mejor AUC-PR en validación y es el candidato final. Su artefacto
  está en `artefactos/modelo_C_logreg.pkl`, y depende de
  `artefactos/modelo_B_gru_pesos.npz` (para el estado oculto del GRU) y
  `artefactos/scaler_agg.pkl` + `artefactos/columnas_agregadas.json` (para las
  variables agregadas). El umbral de decisión elegido (0.075) está en
  `artefactos/parametros_preparacion.json`.
- **Quién usaría el puntaje y qué decisión tomaría:** el equipo de operaciones
  antifraude, en el momento de autorizar una transacción. Un puntaje por encima del
  umbral bloquea la transacción y la envía a revisión manual; por debajo, se autoriza
  automáticamente.
- **Contrato preliminar de entrada/salida:** entrada = las últimas 10 transacciones
  del cliente (ordenadas cronológicamente, con las mismas 14 variables por paso que
  construye la función `_vector_paso` del notebook) más las 13 variables agregadas
  (columna `AGG_COLS` del notebook); salida = un puntaje continuo de riesgo en [0, 1].
  No se especifica todavía un contrato de API/latencia — eso corresponde al Proyecto Final.
- **Límites, riesgos y datos que faltarían:** (1) el dataset es sintético y a escala
  reducida frente a 1.4 millones de tarjetas reales, así que el desempeño reportado
  probablemente esté optimista; (2) el mecanismo `fraude_sigiloso` (sin monto atípico)
  fue, como se esperaba, el más difícil — si el fraude real se parece más a ese
  mecanismo que a los otros cuatro, se necesitaría una señal adicional (por ejemplo,
  verificación de identidad) que este proyecto no incorpora; (3) el umbral fue
  calibrado con los costos hipotéticos del enunciado (Q4,200 / Q180) — antes de producción
  habría que validarlos contra cifras reales del banco; (4) falta evaluar deriva de
  distribución (concept drift) si los patrones de fraude cambian con el tiempo, algo
  que el Proyecto Final sí deberá monitorear.
