
from pathlib import Path
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "2.5g_bnkps.onnx"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

print("SCRFD model loaded successfully!")
print("Providers:", session.get_providers())

print("\nModel inputs:")
for item in session.get_inputs():
    print(item.name, item.shape, item.type)

print("\nModel outputs:")
for item in session.get_outputs():
    print(item.name, item.shape, item.type)
