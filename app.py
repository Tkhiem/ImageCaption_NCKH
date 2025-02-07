import os
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

# Định nghĩa đường dẫn đến mô hình ONNX (model.onnx nằm trong thư mục gốc của project)
MODEL_PATH = os.getenv("MODEL_PATH", "./model.onnx")

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
def get_predict():
    return {"message": "Chức năng dự đoán chưa được triển khai (GET)"}

# Hàm xử lý đầu ra của mô hình
def process_output(ort_outs):
    # Chuyển đầu ra thành NumPy array nếu chưa phải
    if isinstance(ort_outs, list):
        ort_outs = np.array(ort_outs)
    
    print("📌 Shape của output:", ort_outs.shape)
    print("📌 Giá trị đầu ra:", ort_outs)
    print("📌 Kiểu dữ liệu:", ort_outs.dtype)
    
    return ort_outs

# Endpoint POST /predict để nhận file ảnh và trả về chú thích
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # Đọc nội dung file ảnh từ client
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))  # Resize ảnh theo yêu cầu mô hình
        
        # Chuyển đổi ảnh sang mảng NumPy
        input_array = np.array(img).astype(np.float32) / 255.0
        input_array = np.transpose(input_array, (2, 0, 1))  # Chuyển từ (H, W, C) -> (C, H, W)
        input_array = np.expand_dims(input_array, axis=0)  # Thêm batch dimension
        
        # Chuẩn bị input cho mô hình ONNX
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: input_array}

        # Gọi suy luận (inference) của mô hình
        ort_outs = ort_session.run(None, ort_inputs)
        processed_output = process_output(ort_outs[0])
        
        # TODO: Xử lý đầu ra của mô hình để chuyển thành chuỗi chú thích
        caption = "Dummy caption - implement decoding logic here"

        return JSONResponse(content={"caption": caption})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Chạy ứng dụng nếu tệp được chạy trực tiếp
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
