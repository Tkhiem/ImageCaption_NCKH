import json

# Load từ điển từ file
with open("tokenizer/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Kiểm tra nếu key là chuỗi (từ) và value là số (ID) -> Đảo ngược lại
if isinstance(list(vocab.keys())[0], str) and not list(vocab.keys())[0].isdigit():
    vocab = {v: k for k, v in vocab.items()}  # Đảo ngược lại

# Chuyển key về dạng số nguyên
id_to_word = {int(k): v for k, v in vocab.items()}

# Lưu lại từ điển mới (nếu cần)
with open("tokenizer/vocab_fixed.json", "w", encoding="utf-8") as f:
    json.dump(id_to_word, f, ensure_ascii=False, indent=4)

print("✅ Đã sửa vocab.json thành dạng đúng!")
