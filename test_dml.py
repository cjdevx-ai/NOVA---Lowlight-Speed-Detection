import onnxruntime as ort

MODEL_PATH = r"G:\cla_projects\NOVA\yolo8nano_1k_320\best.onnx"

sess = ort.InferenceSession(
    MODEL_PATH,
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)

print("Available providers:", ort.get_available_providers())
print("Session providers:", sess.get_providers())
