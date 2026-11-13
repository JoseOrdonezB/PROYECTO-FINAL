import re
import warnings; warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("../app.py", default_timeout=60).run()
print("exceptions on load:", [e.value for e in at.exception])
def calc(name):
    at.selectbox[0].select(name).run()
    at.button[0].click().run()
    # El resultado se dibuja como HTML: se extrae la recomendación de la tarjeta.
    tarjeta = [m.value for m in at.markdown if "Puntuación de riesgo" in m.value]
    texto = re.search(r"((?:Autorizar|Bloquear) transacción\.[^<]*)", tarjeta[0]).group(1).strip() if tarjeta else None
    print(name, "| exc:", [e.value for e in at.exception], "| errores:", [e.value for e in at.error], "|", texto)
for n in ["Transacción legítima","Transacción fraudulenta (escalada rápida)","Caso límite (zona ámbar)","En blanco"]:
    calc(n)
