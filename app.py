import os
import json
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

# Định nghĩa đường dẫn đến mô hình và từ điển
MODEL_PATH = os.getenv("MODEL_PATH", "./model.onnx")
VOCAB_PATH = os.getenv("VOCAB_PATH", "./tokenizer/vocab_fixed.json")

# Kiểm tra file tồn tại
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy mô hình tại: {MODEL_PATH}")

if not os.path.exists(VOCAB_PATH):
    raise FileNotFoundError(f"Không tìm thấy từ điển tại: {VOCAB_PATH}")

# Load mô hình ONNX
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Lỗi khi load mô hình ONNX: {str(e)}")

# Load từ điển vocab_fixed.json
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Đảm bảo key là `int`, value là `str`
id_to_word = {int(k): v for k, v in vocab.items()}

def decode_tokens(token_ids):
    """Chuyển token ID thành câu caption."""
    if not isinstance(token_ids, list):
        token_ids = [token_ids]

    print("DEBUG - Token IDs nhận được:", token_ids)  # In token_ids trước khi decode

    words = [id_to_word.get(token_id, "[UNK]") for token_id in token_ids]
    print("DEBUG - Ánh xạ từ vựng:", words)  # In danh sách từ được dịch ra

    words = [word for word in words if word not in ["[PAD]", "[START]", "[END]", "[UNK]"]]
    
    caption = " ".join(words).capitalize() + "."
    print("DEBUG - Caption sinh ra:", caption)  # In caption sau khi xử lý
    return caption

@app.get("/")
def read_root():
    return {"message": "API chạy thành công trên Render!"}

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # Đọc ảnh từ client
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))  # Resize ảnh

        # Chuyển ảnh thành NumPy array
        input_array = np.array(img).astype(np.float32) / 255.0
        input_array = np.transpose(input_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        input_array = np.expand_dims(input_array, axis=0)  # Thêm batch dimension

        # Chuẩn bị input cho mô hình
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: input_array}

        # Chạy mô hình
        ort_outs = ort_session.run(None, ort_inputs)

        # Kiểm tra output có đúng định dạng không
        if not isinstance(ort_outs, list) or len(ort_outs) == 0:
            return JSONResponse(status_code=500, content={"error": "Mô hình không trả về output hợp lệ."})

        # Log output gốc của mô hình
        print("DEBUG - Output từ mô hình:", ort_outs[0])

        # Lấy token ID (output của mô hình)
        token_ids = np.argmax(ort_outs[0], axis=-1)  # Lấy ID có xác suất cao nhất
        
        # In token_ids để debug
        print("DEBUG - Token IDs sau khi argmax:", token_ids)

        # Nếu chỉ có 1 caption (batch size = 1), chuyển thành list
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Giải mã thành câu
        caption = decode_tokens(token_ids)

        return JSONResponse(content={"caption": caption})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)} )

# Chạy server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
