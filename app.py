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

# Endpoint POST /predict để nhận file ảnh và trả về chú thích
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Đọc nội dung file ảnh từ client
        contents = await image.read()
        # Mở ảnh bằng Pillow
        img = Image.open(BytesIO(contents))
        img = img.convert("RGB")  # Đảm bảo ảnh ở chế độ RGB

        # Tiền xử lý ảnh: ví dụ, resize về kích thước 224x224 (điều chỉnh theo yêu cầu mô hình)
        img = img.resize((224, 224))
        # Chuyển đổi ảnh sang mảng NumPy kiểu float32
        input_array = np.array(img).astype(np.float32)
        # Chuẩn hóa ảnh (giả sử mô hình yêu cầu giá trị giữa 0 và 1)
        input_array = input_array / 255.0
        # Chuyển đổi định dạng từ (H, W, C) sang (C, H, W) nếu mô hình yêu cầu
        input_array = np.transpose(input_array, (2, 0, 1))
        # Thêm batch dimension: kết quả có shape (1, C, H, W)
        input_array = np.expand_dims(input_array, axis=0)

        # Chuẩn bị input cho mô hình ONNX
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: input_array}

        # Gọi suy luận (inference) của mô hình
        ort_outs = ort_session.run(None, ort_inputs)

        # TODO: Xử lý đầu ra của mô hình để chuyển thành chuỗi chú thích
        # Phần này phụ thuộc vào cấu trúc đầu ra của mô hình của bạn.
        # Ví dụ: nếu đầu ra là một dãy các chỉ số token, bạn cần giải mã chúng thành từ ngữ.
        # Dưới đây là ví dụ giả định:
        caption = "Dummy caption - implement decoding logic here"

        return JSONResponse(content={"caption": caption})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Chạy ứng dụng nếu tệp được chạy trực tiếp
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
