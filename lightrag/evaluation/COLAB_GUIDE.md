# Hướng dẫn chạy đánh giá trên Google Colab

Copy từng cell dưới đây vào Colab theo thứ tự. Đọc kỹ phần **"Bạn cần điền"** trước khi chạy.

---

## CELL 1 — Cài đặt thư viện

```python
# Cài tất cả dependencies cần thiết
!pip install -q uv
!cd /content && git clone https://github.com/HKUDS/LightRAG.git 2>/dev/null || echo "Đã clone rồi"
%cd /content/LightRAG
!pip install -q -e ".[api,test]"
!pip install -q frame-semantic-transformer ragas datasets langchain-openai httpx python-dotenv
```

---

## CELL 2 — Tạo file .env (BẠN CẦN ĐIỀN API KEY)

```python
# ============================================================
# ĐIỀN CÁC GIÁ TRỊ CỦA BẠN VÀO ĐÂY
# ============================================================
LLM_API_KEY      = "sk-..."        # OpenAI API key (cho LightRAG server)
EVAL_API_KEY     = "sk-..."        # OpenAI API key (cho RAGAS, có thể dùng cùng key)
EMBEDDING_MODEL  = "text-embedding-3-large"   # hoặc text-embedding-3-small
LLM_MODEL        = "gpt-4o-mini"              # model LightRAG dùng để index
# ============================================================

# Tự động xác định EMBEDDING_DIM theo model
DIM_MAP = {
    "text-embedding-3-small":  1536,
    "text-embedding-ada-002":  1536,
    "text-embedding-3-large":  3072,
}
EMBEDDING_DIM = DIM_MAP.get(EMBEDDING_MODEL, 1536)
print(f"EMBEDDING_DIM tự động: {EMBEDDING_DIM}")

env_content = f"""# LightRAG server config — tự sinh bởi COLAB_GUIDE
LLM_BINDING=openai
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY={LLM_API_KEY}
LLM_MODEL={LLM_MODEL}

EMBEDDING_BINDING=openai
EMBEDDING_BINDING_HOST=https://api.openai.com/v1
EMBEDDING_BINDING_API_KEY={LLM_API_KEY}
EMBEDDING_MODEL={EMBEDDING_MODEL}
EMBEDDING_DIM={EMBEDDING_DIM}

EVAL_LLM_BINDING_API_KEY={EVAL_API_KEY}
OPENAI_API_KEY={EVAL_API_KEY}

LIGHTRAG_FRAME_EXTRACTION_MODE=full
WORKING_DIR=./rag_storage
MAX_ASYNC=4
MAX_TOKENS=32768
"""

with open("/content/LightRAG/.env", "w") as f:
    f.write(env_content)

print("Đã tạo .env. Kiểm tra:")
with open("/content/LightRAG/.env") as f:
    for line in f:
        if "API_KEY" not in line:          # ẩn key
            print(" ", line.strip())
        else:
            key_name = line.split("=")[0]
            print(f"  {key_name}=****")
```

---

## CELL 3 — Xóa storage cũ (nếu có lỗi từ lần trước)

```python
import shutil, os

storage_path = "/content/LightRAG/rag_storage"   # đây là chỗ đúng
shutil.rmtree(storage_path, ignore_errors=True)
os.makedirs(storage_path, exist_ok=True)
print(f"Storage đã được xóa sạch: {storage_path}")
```

---

## CELL 4 — Khởi động LightRAG server

```python
import subprocess, time, os, sys

os.chdir("/content/LightRAG")

# Kill server cũ nếu có
os.system("pkill -f lightrag_server 2>/dev/null; pkill -f 'lightrag.api' 2>/dev/null; sleep 1")

proc = subprocess.Popen(
    [sys.executable, "-c", "from lightrag.api.lightrag_server import main; main()"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd="/content/LightRAG",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)

print("Đang chờ server khởi động...")
started = False
for i in range(90):          # đợi tối đa 90 giây
    time.sleep(1)
    line = proc.stdout.readline().decode("utf-8", errors="ignore").strip()
    if line:
        print(f"[{i:02d}s] {line}")
    if any(kw in line for kw in ["Uvicorn running", "Application startup complete", "started server"]):
        print("\n✅ Server sẵn sàng!")
        started = True
        break
    if proc.poll() is not None:
        remaining = proc.stdout.read().decode("utf-8", errors="ignore")
        print("❌ Server thoát sớm!\n", remaining[-2000:])
        break

if not started:
    print("⚠️  Chưa thấy thông báo 'ready'. Kiểm tra thủ công:")
    print("  import httpx; print(httpx.get('http://localhost:9621/health').text)")
```

---

## CELL 5 — Kiểm tra server đang chạy

```python
import httpx

try:
    r = httpx.get("http://localhost:9621/health", timeout=5)
    print("✅ Server OK:", r.json())
except Exception as e:
    print("❌ Không kết nối được:", e)
    print("Thử restart lại Cell 4")
```

---

## CELL 6 — Chạy đánh giá đầy đủ (frame-semantic mode=full)

```python
import subprocess, os

os.makedirs("/content/LightRAG/lightrag/evaluation/results", exist_ok=True)

result = subprocess.run(
    [
        "python", "lightrag/evaluation/run_full_eval.py",
        "--output", "lightrag/evaluation/results/eval_full_mode.json",
    ],
    cwd="/content/LightRAG",
    capture_output=True, text=True,
    timeout=900,   # 15 phút tối đa
)

print(result.stderr)
if result.stdout:
    print("STDOUT:", result.stdout)
```

---

## CELL 7 — Chạy mode=none (LLM baseline, để so sánh)

```python
import subprocess, os, shutil

# Xóa storage để index lại từ đầu với mode khác
shutil.rmtree("/content/LightRAG/rag_storage", ignore_errors=True)

# Kill và restart server với mode=none
os.system("pkill -f lightrag_server 2>/dev/null; sleep 2")

import sys, time, subprocess

env_none = {**__import__("os").environ, "LIGHTRAG_FRAME_EXTRACTION_MODE": "none", "PYTHONUNBUFFERED": "1"}
proc2 = subprocess.Popen(
    [sys.executable, "-c", "from lightrag.api.lightrag_server import main; main()"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    cwd="/content/LightRAG", env=env_none,
)
for i in range(90):
    time.sleep(1)
    line = proc2.stdout.readline().decode("utf-8", errors="ignore").strip()
    if line: print(f"[{i:02d}s] {line}")
    if any(kw in line for kw in ["Uvicorn running", "Application startup complete"]):
        print("✅ Server (mode=none) ready!")
        break

# Chạy eval
result2 = subprocess.run(
    [
        "python", "lightrag/evaluation/run_full_eval.py",
        "--output", "lightrag/evaluation/results/eval_none_mode.json",
    ],
    cwd="/content/LightRAG",
    capture_output=True, text=True,
    timeout=900,
    env=env_none,
)
print(result2.stderr[-4000:])
```

---

## CELL 8 — So sánh kết quả 2 mode

```python
import json

def load_ragas(path):
    with open(path) as f:
        data = json.load(f)
    avg = data.get("ragas_eval", {}).get("average_metrics", {})
    return avg

r_full = load_ragas("/content/LightRAG/lightrag/evaluation/results/eval_full_mode.json")
r_none = load_ragas("/content/LightRAG/lightrag/evaluation/results/eval_none_mode.json")

metrics = ["faithfulness", "answer_relevance", "context_recall", "context_precision", "ragas_score"]

print(f"{'Metric':<22}  {'Frame-full':>12}  {'LLM-none':>12}  {'Delta':>10}  Winner")
print("-" * 75)
for m in metrics:
    f = r_full.get(m, 0)
    n = r_none.get(m, 0)
    delta = f - n
    winner = "Frame+" if delta > 0.01 else ("LLM+" if delta < -0.01 else "Hòa")
    print(f"{m:<22}  {f:>12.4f}  {n:>12.4f}  {delta:>+10.4f}  {winner}")
```

---

## Ghi chú quan trọng

### Tại sao RAGAS cho kết quả 0.0?
Nguyên nhân phổ biến (theo thứ tự hay gặp):
1. **EMBEDDING_DIM sai** — phải khớp với model (`text-embedding-3-large` → 3072)
2. **Server chưa sẵn sàng khi index** — đợi Cell 5 OK rồi mới chạy Cell 6
3. **Storage chưa được xóa** — chạy Cell 3 trước khi restart server
4. **Documents failed** — xem log "Trang thai:" trong output Cell 6

### Chi phí ước tính (1 lần chạy đầy đủ 2 mode)
| Thành phần | Chi phí ước tính |
|-----------|-----------------|
| Indexing 5 docs × 2 modes (LLM+embedding) | ~$0.05–0.15 |
| RAGAS 6 queries × 2 modes | ~$0.02–0.05 |
| **Tổng** | **~$0.10–0.20** |

> Dùng `gpt-4o-mini` cho LLM và `text-embedding-3-small` (1536 dim) sẽ rẻ hơn nhiều.
