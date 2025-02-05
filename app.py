import os
import onnxruntime as ort
from fastapi import FastAPI

app = FastAPI()

# Định nghĩa đường dẫn đến mô hình (Render sử dụng thư mục /opt/render để lưu file)
MODEL_DIR = os.getenv("MODEL_DIR", "./model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")

# Kiểm tra xem mô hình có tồn tại không
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy mô hình tại: {MODEL_PATH}")

# Load mô hình ONNX
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Lỗi khi load mô hình ONNX: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "API chạy thành công trên Render!"}

@app.get("/predict")
def predict():
    return {"message": "Chức năng dự đoán chưa được triển khai"}

# Đoạn này giúp Render chạy uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
