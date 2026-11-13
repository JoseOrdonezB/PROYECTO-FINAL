"""
Sube la app a un Space de Hugging Face (SDK Docker + Streamlit).

Hugging Face ya no ofrece Streamlit como SDK directo (abril 2025): ahora un Space de
Streamlit es un Space Docker. El Dockerfile de la raíz instala requirements.txt y
levanta streamlit en el puerto 8501.

Uso (desde la raíz del repo, con el token de HF en la variable HF_TOKEN):
    pip install huggingface_hub
    export HF_TOKEN=hf_xxx            # Windows PowerShell: $env:HF_TOKEN="hf_xxx"
    python despliegue/subir_a_hf.py TU_USUARIO/monitoreo-fraude

Sube: Dockerfile, app.py, scoring.py, requirements.txt, artefactos/ y el README
(con la configuración del Space). Imprime la URL pública al terminar.
"""
import os
import sys

from huggingface_hub import HfApi

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("Uso: python despliegue/subir_a_hf.py USUARIO/NOMBRE_DEL_SPACE")
    repo_id = sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Falta la variable de entorno HF_TOKEN (token con permiso de escritura).")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    for archivo in ["Dockerfile", "app.py", "scoring.py", "requirements.txt"]:
        api.upload_file(path_or_fileobj=os.path.join(RAIZ, archivo),
                        path_in_repo=archivo, repo_id=repo_id, repo_type="space")

    api.upload_file(path_or_fileobj=os.path.join(RAIZ, "despliegue", "README.md"),
                    path_in_repo="README.md", repo_id=repo_id, repo_type="space")

    api.upload_folder(folder_path=os.path.join(RAIZ, "artefactos"),
                      path_in_repo="artefactos", repo_id=repo_id, repo_type="space")

    usuario, nombre = repo_id.split("/")
    directo = f"https://{usuario}-{nombre}.hf.space".lower().replace("_", "-").replace(".", "-")
    print(f"Listo. Página del Space : https://huggingface.co/spaces/{repo_id}")
    print(f"       App directa      : {directo}")
    print("Espere 3-5 minutos a que el Space termine de construirse (pestaña 'Logs').")


if __name__ == "__main__":
    main()
