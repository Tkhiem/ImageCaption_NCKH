import os
import onnxruntime as ort
from fastapi import FastAPI

app = FastAPI()

# Định nghĩa đường dẫn đến mô hìn
model_path = os.path.join(os.getcwd(), "model", "model.onnx")

# Kiểm tra xem mô hình có tồn tại không
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_path}")

# Load mô hình ONNX
try:
    ort_session = ort.InferenceSession(model_path)
except Exception as e:
    raise RuntimeError(f"Lỗi khi load mô hình ONNX: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "API chạy thành công!"}

@app.get("/predict")
def predict():
    return {"message": "Chức năng dự đoán chưa được triển khai"}
