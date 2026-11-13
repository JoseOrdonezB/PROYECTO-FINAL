import warnings; warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("../app.py", default_timeout=60).run()
print("exceptions on load:", [e.value for e in at.exception])
def calc(name):
    at.selectbox[0].select(name).run()
    at.button[0].click().run()
    out = [e.value for e in at.error] + [m.label+"="+str(m.value) for m in at.metric]
    print(name, "| exc:", [e.value for e in at.exception], "|", out, "|", [s.value for s in at.success][:1])
for n in ["Transacción legítima","Transacción fraudulenta (escalada rápida)","Caso límite (zona ámbar)","En blanco"]:
    calc(n)
