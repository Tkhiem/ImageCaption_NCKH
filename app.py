import os
import json
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

# Định nghĩa đường dẫn đến mô hình ONNX (model.onnx nằm trong thư mục gốc của project)
MODEL_PATH = os.getenv("MODEL_PATH", "./model.onnx")
VOCAB_PATH = os.getenv("VOCAB_PATH", "./tokenizer/vocab.json")

# Kiểm tra xem mô hình có tồn tại không
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy mô hình tại: {MODEL_PATH}")

# Load mô hình ONNX
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Lỗi khi load mô hình ONNX: {str(e)}")

# Load từ điển vocab.json
if not os.path.exists(VOCAB_PATH):
    raise FileNotFoundError(f"Không tìm thấy file từ điển tại: {VOCAB_PATH}")

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Tạo ánh xạ ID -> từ
id_to_word = {int(k): v for k, v in vocab.items()}

def decode_tokens(token_ids):
    """Chuyển token ID thành câu caption."""
    words = [id_to_word.get(token_id, "") for token_id in token_ids]
    words = [word for word in words if word not in ["[PAD]", "[START]", "[END]"]]  # Loại bỏ token đặc biệt
    return " ".join(words).capitalize() + "."

@app.get("/")
def read_root():
    return {"message": "API chạy thành công trên Render!"}

@app.get("/predict")
def get_predict():
    return {"message": "Chức năng dự đoán chưa được triển khai (GET)"}

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
        
        # Lấy token ID sau argmax
        token_ids = np.argmax(ort_outs[0], axis=-1).tolist()

        # Giải mã token ID thành caption
        caption = decode_tokens(token_ids)

        return JSONResponse(content={"caption": caption})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Chạy ứng dụng nếu tệp được chạy trực tiếp
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
