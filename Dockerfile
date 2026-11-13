# Imagen del Space de Hugging Face (SDK Docker + Streamlit).
# Hugging Face corre el contenedor con el usuario 1000, por eso se crea aquí.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app.py scoring.py ./
COPY --chown=user artefactos ./artefactos

EXPOSE 8501
# enableXsrfProtection=false: sin esto el proxy de HF bloquea la subida de CSV (error 403)
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false"]
