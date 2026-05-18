# LightRAG — Tích hợp Frame-Semantic Transformer (3 Chế Độ)

Tài liệu này mô tả **toàn bộ thay đổi**, **kiến trúc 3 chế độ**, **cách cài đặt**, **cách triển khai**, và **cách chạy đánh giá** của phiên bản LightRAG đã được sửa đổi để sử dụng thư viện [`frame-semantic-transformer`](https://github.com/chanind/frame-semantic-transformer) (tác giả: chanind).

---

## Mục lục

1. [Tổng quan thay đổi](#1-tổng-quan-thay-đổi)
2. [Ba chế độ trích xuất](#2-ba-chế-độ-trích-xuất)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Ánh xạ ngữ nghĩa Frame → LightRAG](#4-ánh-xạ-ngữ-nghĩa-frame--lightrag)
5. [Danh sách các file đã thay đổi / thêm mới](#5-danh-sách-các-file-đã-thay-đổi--thêm-mới)
6. [Yêu cầu hệ thống](#6-yêu-cầu-hệ-thống)
7. [Xung đột phụ thuộc (Protobuf)](#7-xung-đột-phụ-thuộc-protobuf)
8. [Cài đặt — Môi trường Conda (Khuyến nghị)](#8-cài-đặt--môi-trường-conda-khuyến-nghị)
9. [Cài đặt — Virtualenv](#9-cài-đặt--virtualenv)
10. [Cấu hình `.env`](#10-cấu-hình-env)
11. [Chạy nhanh (Quick Start)](#11-chạy-nhanh-quick-start)
12. [Chạy API Server](#12-chạy-api-server)
13. [Chạy Tests](#13-chạy-tests)
14. [Đánh giá & So sánh (Evaluation)](#14-đánh-giá--so-sánh-evaluation)
15. [Hướng dẫn so sánh 3 chế độ đầy đủ](#15-hướng-dẫn-so-sánh-3-chế-độ-đầy-đủ)
16. [Luồng xử lý chi tiết](#16-luồng-xử-lý-chi-tiết)
17. [Câu hỏi thường gặp](#17-câu-hỏi-thường-gặp)

---

## 1. Tổng quan thay đổi

### Vấn đề gốc

Phiên bản LightRAG gốc sử dụng LLM cho **hai mục đích trích xuất ngữ nghĩa**:

| Bước | Giai đoạn | Mô tả |
|------|-----------|-------|
| Offline indexing | Xây đồ thị | LLM phân tích từng đoạn văn bản → trích xuất thực thể và quan hệ |
| Online retrieval | Tìm kiếm | LLM phân tích câu truy vấn → trích xuất từ khóa high-level & low-level |

**Nhược điểm:** Tốn chi phí API, bị giới hạn rate limit, phụ thuộc vào mạng internet, và có độ trễ cao.

### Giải pháp

Thay thế các bước trích xuất ngữ nghĩa bằng `frame-semantic-transformer` — một mô hình **T5** được fine-tune trên **FrameNet** của Berkeley, **hoàn toàn chạy cục bộ, không cần API key, không tốn chi phí**.

```
Trước:  Văn bản → [LLM prompt + API call] → JSON → entities/keywords
Sau:    Văn bản → [T5 FrameNet model cục bộ] → Frames + FEs → entities/keywords
```

**Lưu ý quan trọng:** Các pha **de-duplicate** và **summarise** (tổng hợp mô tả thực thể/quan hệ) **vẫn dùng LLM** ở cả 3 chế độ — chỉ pha trích xuất ngữ nghĩa ban đầu mới thay đổi.

---

## 2. Ba chế độ trích xuất

Phiên bản này hỗ trợ 3 chế độ hoạt động, điều chỉnh qua biến môi trường:

```bash
export LIGHTRAG_FRAME_EXTRACTION_MODE=full   # hoặc none, hl_only
```

### Mode `none` — Baseline (LightRAG gốc)

**Toàn bộ trích xuất dùng LLM.** Giống hệt phiên bản LightRAG gốc.

| Giai đoạn | Trích xuất |
|-----------|-----------|
| Offline: entities | LLM (prompt + gleaning) |
| Offline: relations | LLM (từ cùng prompt) |
| Online: hl_keywords | LLM (`keywords_extraction` prompt) |
| Online: ll_keywords | LLM (`keywords_extraction` prompt) |
| De-duplicate/Summarise | **LLM** (không thay đổi) |

### Mode `hl_only` — Case 1 (Frame-semantic cho HL)

**Frame-semantic cho high-level keywords; LLM cho low-level và entities.**

| Giai đoạn | Trích xuất |
|-----------|-----------|
| Offline: entities | **LLM** (prompt + gleaning) |
| Offline: relations | **LLM** (từ cùng prompt) |
| Online: hl_keywords | **Frame-semantic** (tên frame) |
| Online: ll_keywords | **LLM** (`keywords_extraction` prompt, chỉ lấy ll part) |
| De-duplicate/Summarise | **LLM** (không thay đổi) |

### Mode `full` — Case 2 (Frame-semantic + LLM normalize) ← **Mặc định**

**Frame-semantic trích xuất, LLM chuẩn hóa/lọc noise, rồi mới indexing.**

| Giai đoạn | Trích xuất |
|-----------|-----------|
| Offline: entities (raw) | **Frame-semantic** (FE texts → entity nodes) |
| Offline: entities (clean) | **LLM filter** — loại noise, chuẩn hóa tên |
| Offline: relations | **Frame-semantic** → cập nhật theo tên đã normalize |
| Online: hl_keywords (raw) | **Frame-semantic** (tên frame) |
| Online: hl_keywords (clean) | **LLM filter** — loại frame không liên quan |
| Online: ll_keywords (raw) | **Frame-semantic** (FE texts) |
| Online: ll_keywords (clean) | **LLM filter** — loại thời gian, số, danh từ chung |
| De-duplicate/Summarise | **LLM** (không thay đổi) |

### Bảng so sánh tổng hợp

```
┌──────────────────────────────┬───────────┬──────────┬────────────────────┐
│ Thành phần                   │  none     │ hl_only  │        full        │
├──────────────────────────────┼───────────┼──────────┼────────────────────┤
│ Entity extraction (offline)  │    LLM    │   LLM    │ Frame → LLM clean  │
│ Relation extraction (offline)│    LLM    │   LLM    │ Frame → LLM clean  │
│ HL keyword (online)          │    LLM    │  Frame   │ Frame → LLM filter │
│ LL keyword (online)          │    LLM    │   LLM    │ Frame → LLM filter │
│ De-duplicate / Summarise     │    LLM    │   LLM    │        LLM         │
├──────────────────────────────┼───────────┼──────────┼────────────────────┤
│ Cần LLM API key?             │    Có     │    Có    │         Có*        │
│ Chi phí LLM (indexing)       │    Cao    │   Cao    │     Trung bình     │
│ Tốc độ indexing              │   Chậm    │  Chậm    │      Vừa phải      │
└──────────────────────────────┴───────────┴──────────┴────────────────────┘
* Full mode dùng LLM nhẹ (filter/normalize) thay vì LLM nặng (extract từ đầu)
```

---

## 3. Kiến trúc hệ thống

```
lightrag/
├── frame_extractor.py        ← MỚI: module trung tâm, wrap frame-semantic-transformer
├── operate.py                ← SỬA: 3-mode dispatch cho entity và keyword extraction
│
lightrag/evaluation/
├── eval_frame_semantic.py    ← MỚI: đánh giá chất lượng frame extraction + RAGAS
├── run_comparison.py         ← MỚI: so sánh frame-semantic vs LLM (2 chế độ)
├── run_three_mode_eval.py    ← MỚI: so sánh cả 3 chế độ
├── run_full_eval.py          ← MỚI: pipeline đánh giá đầy đủ
└── sample_dataset.json       ← MỚI: dataset mẫu cho RAGAS evaluation
│
tests/
└── test_extract_entities.py  ← VIẾT LẠI: test mới cho frame-semantic pipeline
│
pyproject.toml                ← SỬA: thêm extras group [frame-semantic]
FRAME_SEMANTIC_README.md      ← MỚI: tài liệu này
```

### `frame_extractor.py` — Module trung tâm

```python
# Async interface — offload T5 inference vào thread executor
await detect_frames(text)                              # → ExtractedFrameCollection

# Dùng cho ONLINE RETRIEVAL (query → keywords)
await extract_keywords_from_frames(text)               # → (hl_keywords, ll_keywords)

# Dùng cho OFFLINE INDEXING (chunk → graph nodes/edges)
await extract_entities_from_frames(text, chunk_key, file_path)  # → (nodes, edges)
```

Model T5 được load **một lần duy nhất** qua `@lru_cache` singleton, dùng lại cho tất cả các lời gọi tiếp theo.

### `operate.py` — Các hàm chính đã thay đổi

```python
# Biến môi trường kiểm soát chế độ
FRAME_EXTRACTION_MODE = os.getenv("LIGHTRAG_FRAME_EXTRACTION_MODE", "full")

# Hàm nội bộ mới: keyword extraction thuần LLM (dùng bởi mode "none" và "hl_only")
async def _extract_keywords_with_llm(text, param, global_config, language)
    → (hl_keywords, ll_keywords)

# Hàm đã cập nhật: dispatch theo mode
async def extract_keywords_only(text, param, global_config, hashing_kv)
    → (hl_keywords, ll_keywords)

# Hàm đã cập nhật: entity extraction với LLM fallback cho mode "none"/"hl_only"
async def extract_entities(chunks, global_config, ...)
```

---

## 4. Ánh xạ ngữ nghĩa Frame → LightRAG

### Frame-Semantic là gì?

FrameNet là cơ sở dữ liệu ngôn ngữ học mô tả **các khung ngữ nghĩa (frames)** — tình huống điển hình trong ngôn ngữ.

```
Câu: "The woman sold the car to the man for five thousand dollars."

Frame: Commerce_sell
  └─ Seller:  "The woman"         → entity_type = "seller"
  └─ Goods:   "the car"           → entity_type = "goods"
  └─ Buyer:   "the man"           → entity_type = "buyer"
  └─ Money:   "five thousand dollars" → entity_type = "money"
```

### Ánh xạ vào LightRAG

| Khái niệm Frame-Semantic | Khái niệm LightRAG | Giai đoạn |
|--------------------------|---------------------|-----------|
| **Tên Frame** (`Commerce_sell`) | **high-level keyword** | Online retrieval |
| **Frame Element text** (`"The woman"`) | **low-level keyword** | Online retrieval |
| **Frame Element** → entity node | `entity_name=FE text`, `entity_type=FE role` | Offline indexing |
| **Cặp FE trong cùng frame** → edge | `keywords=frame name` | Offline indexing |

### Ví dụ thực tế

**Input:** `"Apple CEO Tim Cook announced the new iPhone 16 at the annual event."`

**Frame-semantic phân tích:**
```
Frame: Statement
  └─ Speaker:    "Apple CEO Tim Cook"
  └─ Message:    "the new iPhone 16"

Frame: Announcing
  └─ Communicator: "Apple CEO Tim Cook"
  └─ Information:  "the new iPhone 16"
  └─ Place:        "the annual event"
```

**Kết quả trong LightRAG:**

*Nodes:*
```
"Apple CEO Tim Cook"   [entity_type: speaker / communicator]
"the new iPhone 16"    [entity_type: message / information]
"the annual event"     [entity_type: place]
```

*Edges:*
```
("Apple CEO Tim Cook", "the new iPhone 16")   keywords="Statement"
("Apple CEO Tim Cook", "the annual event")    keywords="Announcing"
```

*Online keywords:*
```
hl_keywords = ["Statement", "Announcing"]
ll_keywords = ["Apple CEO Tim Cook", "the new iPhone 16", "the annual event"]
```

---

## 5. Danh sách các file đã thay đổi / thêm mới

### File MỚI

| File | Mô tả |
|------|-------|
| `lightrag/frame_extractor.py` | Module trung tâm wrap frame-semantic-transformer |
| `lightrag/evaluation/eval_frame_semantic.py` | Đánh giá nội bộ frame extraction + RAGAS end-to-end |
| `lightrag/evaluation/run_comparison.py` | Script so sánh frame-semantic vs LLM (binary) |
| `lightrag/evaluation/run_three_mode_eval.py` | Script so sánh cả 3 chế độ |
| `lightrag/evaluation/run_full_eval.py` | Pipeline đánh giá đầy đủ 4 bước |
| `lightrag/evaluation/sample_dataset.json` | Dataset mẫu cho RAGAS |
| `FRAME_SEMANTIC_README.md` | Tài liệu này |

### File ĐÃ SỬA

| File | Thay đổi chính |
|------|----------------|
| `lightrag/operate.py` | Thêm `FRAME_EXTRACTION_MODE`, `_extract_keywords_with_llm()`, cập nhật `extract_keywords_only()` và `extract_entities()` với 3-mode dispatch |
| `tests/test_extract_entities.py` | Viết lại 3 test cases cho frame-semantic pipeline |
| `pyproject.toml` | Thêm extras group `[frame-semantic]` (tách riêng khỏi main deps để tránh xung đột protobuf) |

### Chi tiết thay đổi `operate.py`

```python
# 1. Import mới (dòng ~16-58)
from lightrag.utils import (..., update_chunk_cache_list, remove_think_tags, pack_user_ass_to_openai_messages, ...)
from lightrag.frame_extractor import extract_keywords_from_frames, extract_entities_from_frames

# 2. Biến module-level mới (dòng ~87)
FRAME_EXTRACTION_MODE: str = os.getenv("LIGHTRAG_FRAME_EXTRACTION_MODE", "full").lower().strip()

# 3. Các hàm private mới (theo thứ tự trong file)
async def _llm_filter_frame_entities(maybe_nodes, maybe_edges, global_config, llm_response_cache):
    """Filter + normalize frame-extracted entities via LLM. Dùng sau extract_entities_from_frames()."""
    # Prompt: frame_entity_filter — LLM trả về {"entities": [{"original": "...", "normalized": "..."}, ...]}
    # → loại noise, chuẩn hóa tên, cập nhật edges theo tên mới

async def _llm_filter_frame_keywords(hl_keywords, ll_keywords, global_config, param):
    """Filter + clean frame-extracted keywords via LLM. Dùng sau extract_keywords_from_frames()."""
    # Prompt: frame_keyword_filter — LLM trả về {"high_level_keywords": [...], "low_level_keywords": [...]}

async def _extract_keywords_with_llm(text, param, global_config, language):
    """LLM-based keyword extraction (original LightRAG). Dùng bởi mode 'none' và 'hl_only'."""

# 4. extract_keywords_only() — dispatch theo mode:
#    - none    → _extract_keywords_with_llm() cho cả hl và ll
#    - hl_only → extract_keywords_from_frames() → hl; _extract_keywords_with_llm() → ll
#    - full    → extract_keywords_from_frames() → (hl, ll) → _llm_filter_frame_keywords()

# 5. extract_entities() → _process_single_content() — dispatch theo mode:
#    - full         → extract_entities_from_frames() → _llm_filter_frame_entities()
#    - none/hl_only → LLM entity extraction (với gleaning, giống code gốc)
```

---

## 6. Yêu cầu hệ thống

| Yêu cầu | Phiên bản |
|---------|-----------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0.0 (CPU hoặc CUDA) |
| RAM | Tối thiểu 4 GB (model T5-base ~850 MB) |
| Disk | ~1.5 GB cho model weights (HuggingFace cache) |
| GPU | Không bắt buộc (CPU đủ chạy, chậm hơn ~5-10x) |

**Phụ thuộc Python của `frame-semantic-transformer`:**

| Package | Phiên bản yêu cầu |
|---------|------------------|
| `frame-semantic-transformer` | ≥ 0.1.0, < 1.0.0 |
| `torch` | ≥ 2.0.0 |
| `transformers` | ≥ 4.18.0, < 5.0.0 |
| `pytorch-lightning` | ≥ 1.6.2, < 2.0.0 |
| `sentencepiece` | ≥ 0.1.97, < 0.2.0 |
| `protobuf` | ≥ 3.20.1, **< 4.0.0** ⚠️ |
| `nlpaug` | ≥ 1.1.11, < 2.0.0 |
| `nltk` | ≥ 3.7, < 4.0 |
| `tqdm` | ≥ 4.64.0 |

---

## 7. Xung đột phụ thuộc (Protobuf)

### Vấn đề

`frame-semantic-transformer` yêu cầu `protobuf < 4.0.0`, trong khi:
- `google-genai` (Google Gemini provider) yêu cầu `protobuf >= 4.x`
- Hai thư viện này **không thể cài cùng nhau** trong một môi trường Python

### Giải pháp đã áp dụng

`frame-semantic-transformer` đã được **tách ra khỏi main dependencies** và đưa vào extras group riêng trong `pyproject.toml`:

```toml
[project.optional-dependencies]
frame-semantic = [
    "frame-semantic-transformer>=0.1.0,<1.0.0",
    "protobuf>=3.20.1,<4.0.0",
    "torch>=2.0.0",
    ...
]
```

### Cách cài đặt không xung đột

**Cách 1 — Conda (khuyến nghị):** Tạo môi trường riêng với `protobuf=3.x` (xem mục 8).

**Cách 2 — Virtualenv riêng:** Không cài `google-genai` (xem mục 9).

**Cách 3 — Chạy song song:** Dùng `none` mode (LLM thuần) trong môi trường chính, dùng `full` mode trong môi trường frame-semantic riêng.

---

## 8. Cài đặt — Môi trường Conda (Khuyến nghị)

```bash
# 1. Tạo môi trường mới với Python 3.11 và protobuf 3.x
conda create -n lightrag-frame python=3.11 protobuf=3.20.3 -c conda-forge -y
conda activate lightrag-frame

# 2. Cài PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Cài LightRAG với frame-semantic extras
cd D:/LightRAG
pip install -e ".[api,frame-semantic]"

# 4. (Lần đầu) Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# 5. Kiểm tra cài đặt
python -c "from frame_semantic_transformer import FrameSemanticTransformer; print('OK')"
```

**Với GPU CUDA:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 9. Cài đặt — Virtualenv

```bash
# Tạo venv (Python 3.11 khuyến nghị)
python -m venv .venv-frame
source .venv-frame/bin/activate     # Linux/Mac
# hoặc: .venv-frame\Scripts\activate  # Windows

# Cài LightRAG
pip install -e ".[api,frame-semantic]"

# KHÔNG cài google-genai (xung đột protobuf)
# Nếu cần Gemini, tạo venv riêng hoặc dùng conda
```

---

## 10. Cấu hình `.env`

Sao chép file mẫu và điền thông tin:

```bash
cp env.example .env
```

**Các biến liên quan đến frame-semantic mode:**

```dotenv
# ───── Chế độ trích xuất ngữ nghĩa ─────────────────────────────────────────
# Giá trị: none | hl_only | full
# - none    : Baseline — LLM cho mọi thứ (giống LightRAG gốc)
# - hl_only : Frame-semantic → HL keywords; LLM → LL keywords + entities
# - full    : Frame-semantic → cả HL lẫn LL + entities (mặc định)
LIGHTRAG_FRAME_EXTRACTION_MODE=full

# ───── Cấu hình LLM cơ bản (vẫn cần cho de-duplicate/summarise) ──────────
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
LLM_BINDING_API_KEY=sk-...

# ───── Embedding (cần cho vector search) ──────────────────────────────────
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_BINDING_API_KEY=sk-...

# ───── Cấu hình RAGAS evaluation (tùy chọn) ──────────────────────────────
EVAL_LLM_BINDING_API_KEY=sk-...   # hoặc OPENAI_API_KEY
EVAL_LLM_MODEL=gpt-4o-mini
EVAL_QUERY_MODE=hybrid
LIGHTRAG_API_URL=http://localhost:9621
```

---

## 11. Chạy nhanh (Quick Start)

### Mode `full` (mặc định, không cần LLM cho extraction)

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
import os

os.environ["LIGHTRAG_FRAME_EXTRACTION_MODE"] = "full"

async def main():
    rag = LightRAG(
        working_dir="./storage",
        llm_model_func=gpt_4o_mini_complete,  # vẫn cần cho summarise
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()

    # Indexing — frame-semantic thay LLM cho entity extraction
    await rag.ainsert("The woman sold the car to the man for five thousand dollars.")

    # Query — frame-semantic thay LLM cho keyword extraction
    result = await rag.aquery(
        "Who sold what to whom?",
        param=QueryParam(mode="hybrid")
    )
    print(result)
    await rag.finalize_storages()

asyncio.run(main())
```

### Chuyển đổi chế độ bằng env var

```bash
# Baseline (LLM thuần)
LIGHTRAG_FRAME_EXTRACTION_MODE=none python your_script.py

# Case 1: Frame chỉ cho HL
LIGHTRAG_FRAME_EXTRACTION_MODE=hl_only python your_script.py

# Case 2: Frame cho cả hai (mặc định)
LIGHTRAG_FRAME_EXTRACTION_MODE=full python your_script.py
```

---

## 12. Chạy API Server

```bash
# Đặt mode trước khi chạy server
export LIGHTRAG_FRAME_EXTRACTION_MODE=full   # Linux/Mac
# hoặc: $env:LIGHTRAG_FRAME_EXTRACTION_MODE = "full"  # PowerShell

# Chạy server (port 9621)
lightrag-server

# Hoặc dev mode với reload
uvicorn lightrag.api.lightrag_server:app --reload --port 9621
```

**Lưu ý:** Lần đầu tiên server nhận request indexing/query, T5 model sẽ được download từ HuggingFace (~850 MB) và load vào RAM. Quá trình này mất 1-3 phút. Các lần sau dùng cache.

---

## 13. Chạy Tests

```bash
# Chạy test frame-semantic (offline, không cần server/API key)
python -m pytest tests/test_extract_entities.py -v

# Chạy toàn bộ offline tests
python -m pytest tests/ -v

# Kết quả kỳ vọng:
# tests/test_extract_entities.py::test_extract_entities_calls_frame_extractor PASSED
# tests/test_extract_entities.py::test_extract_entities_processes_multiple_chunks PASSED
# tests/test_extract_entities.py::test_extract_entities_returns_empty_on_frame_failure PASSED
```

---

## 14. Đánh giá & So sánh (Evaluation)

### Cấu trúc thư mục evaluation

```
lightrag/evaluation/
├── eval_frame_semantic.py     # Module đánh giá chính
├── run_comparison.py          # So sánh frame-semantic vs LLM (2 chế độ)
├── run_three_mode_eval.py     # So sánh 3 chế độ
├── run_full_eval.py           # Pipeline đánh giá đầy đủ
├── sample_dataset.json        # Dataset mẫu Q&A
└── results/                   # Kết quả evaluation (tự tạo)
```

### Script 1: Đánh giá nội bộ frame extraction

```bash
# Chạy nhanh — kiểm tra frame detection (không cần server)
python lightrag/evaluation/eval_frame_semantic.py --internal-only
```

**Đầu ra gồm:** Frame recall, Frame precision, Entity recall, tốc độ trung bình.

### Script 2: Đánh giá RAGAS end-to-end

```bash
# Cần: LightRAG server đang chạy + OPENAI_API_KEY

# Đánh giá chế độ hiện tại (theo LIGHTRAG_FRAME_EXTRACTION_MODE trong .env)
python lightrag/evaluation/eval_frame_semantic.py

# So sánh với kết quả LLM cũ
python lightrag/evaluation/eval_frame_semantic.py \
    --compare lightrag/evaluation/results/ragas_none_20250518.json
```

### Script 3: Keyword extraction — 3 chế độ

```bash
# Đánh giá keyword extraction (frame modes, không cần LLM hay server)
python lightrag/evaluation/run_three_mode_eval.py --keyword-only

# So sánh với kết quả RAGAS đã thu thập
python lightrag/evaluation/run_three_mode_eval.py \
    --ragas-files \
        none=results/ragas_none.json \
        hl_only=results/ragas_hlonly.json \
        full=results/ragas_full.json
```

### Script 4: Pipeline đầy đủ

```bash
# Chạy toàn bộ 4 bước (cần server + API key)
python lightrag/evaluation/run_full_eval.py

# Chỉ kiểm tra frame extraction (bước 1-2)
python lightrag/evaluation/run_full_eval.py --internal-only

# So sánh nhanh (bỏ qua indexing)
python lightrag/evaluation/run_full_eval.py --skip-index
```

---

## 15. Hướng dẫn so sánh 3 chế độ đầy đủ

Để so sánh chính xác, cần build **3 index riêng biệt** (một cho mỗi chế độ) rồi chạy RAGAS trên từng index. Các bước:

### Bước 1 — Build index Mode `none` (LLM baseline)

```bash
# Tạo thư mục riêng cho baseline
mkdir -p ./storage_none

# Chạy server với mode none và storage riêng
LIGHTRAG_FRAME_EXTRACTION_MODE=none \
WORKING_DIR=./storage_none \
lightrag-server &

# Đẩy documents vào index
python lightrag/evaluation/run_full_eval.py \
    --internal-only \     # bỏ qua RAGAS, chỉ index
    --docs-dir your_docs/

# Chạy RAGAS eval và lưu kết quả
python lightrag/evaluation/eval_frame_semantic.py \
    --ragas-only \
    --output results/ragas_none.json

kill %1  # tắt server
```

### Bước 2 — Build index Mode `hl_only`

```bash
mkdir -p ./storage_hlonly

LIGHTRAG_FRAME_EXTRACTION_MODE=hl_only \
WORKING_DIR=./storage_hlonly \
lightrag-server &

# Index + eval tương tự...
python lightrag/evaluation/eval_frame_semantic.py --ragas-only --output results/ragas_hlonly.json

kill %1
```

### Bước 3 — Build index Mode `full`

```bash
mkdir -p ./storage_full

LIGHTRAG_FRAME_EXTRACTION_MODE=full \
WORKING_DIR=./storage_full \
lightrag-server &

# Index + eval tương tự...
python lightrag/evaluation/eval_frame_semantic.py --ragas-only --output results/ragas_full.json

kill %1
```

### Bước 4 — So sánh 3 chế độ

```bash
python lightrag/evaluation/run_three_mode_eval.py \
    --keyword-only \
    --ragas-files \
        none=results/ragas_none.json \
        hl_only=results/ragas_hlonly.json \
        full=results/ragas_full.json
```

**Đầu ra mẫu:**

```
======================================================================
SO SANH 3 CHE DO RAGAS
======================================================================
Metric                      none        hl_only        full  Best
----------------------------------------------------------------------
faithfulness              0.8234         0.7891      0.8012  none
answer_relevance          0.7456         0.7823      0.8102  full
context_recall            0.6934         0.7234      0.7456  full
context_precision         0.7123         0.7445      0.7789  full
ragas_score               0.7437         0.7598      0.7840  full
======================================================================
So chi tieu thang: none=1 | hl_only=0 | full=4
```

---

## 16. Luồng xử lý chi tiết

### Offline Indexing (Xây đồ thị)

```
Document
    │
    ▼ chunking_by_token_size()
Chunks
    │
    ├─[mode == "full"] ─────────────────────────────────────────────────────┐
    │                                                                        │
    │  1. extract_entities_from_frames(chunk_text)   ← T5 model             │
    │     ├── detect_frames() → Frames + Frame Elements                     │
    │     ├── FE texts → maybe_nodes (raw, có thể có noise)                 │
    │     └── FE pairs → maybe_edges                                        │
    │                                                                        │
    │  2. _llm_filter_frame_entities(maybe_nodes, maybe_edges)  ← LLM nhẹ  │
    │     ├── Loại bỏ: thời gian, số, đại từ, danh từ chung                 │
    │     ├── Chuẩn hóa tên: "the car" → "car", fix casing                  │
    │     ├── Merge trùng: cùng thực thể, tên khác nhau → 1 node            │
    │     └── Cập nhật src_id/tgt_id trong edges theo tên mới               │
    │                                                                        │
    └─[mode in {"none", "hl_only"}] ────────────────────────────────────────┤
       │                                                                     │
       │  LLM entity extraction (prompt + gleaning)                         │
       │  ├── entity_extraction_system/user_prompt → LLM nặng               │
       │  ├── _process_extraction_result() → maybe_nodes                    │
       │  └── (gleaning round nếu entity_extract_max_gleaning > 0)          │
       │                                                                     │
       └─────────────────────────────────────────────────────────────────────┘
    │
    ▼ (luôn dùng LLM, cả 3 mode)
_handle_entity_relation_summary()  ← LLM de-duplicate + summarise descriptions
    │
    ▼
Knowledge Graph + Vector DB
```

### Online Retrieval (Truy xuất)

```
Query
    │
    ▼ extract_keywords_only()
    │
    ├─[none]──► _extract_keywords_with_llm() → (hl, ll)
    │           └── keywords_extraction prompt → LLM
    │
    ├─[hl_only]─► extract_keywords_from_frames() → hl    (T5)
    │             _extract_keywords_with_llm() → ll       (LLM)
    │
    └─[full]──► extract_keywords_from_frames() → raw (hl, ll)   (T5)
                _llm_filter_frame_keywords() → clean (hl, ll)    (LLM nhẹ)
                ├── hl: giữ frame names có nghĩa, loại nonsense
                └── ll: loại thời gian, số, đại từ, danh từ chung
    │
    ▼
hl_keywords → query relations VDB
ll_keywords → query entities VDB
    │
    ▼
Build context → LLM generate answer
```

---

## 17. Câu hỏi thường gặp

### Q: Model T5 download ở đâu, bao lâu?

**A:** Model được download tự động từ HuggingFace (`~850 MB`) vào thư mục cache (`~/.cache/huggingface/`). Chỉ download lần đầu, các lần sau dùng cache. Trên internet thường: 2-10 phút tùy tốc độ mạng.

### Q: Chế độ nào nên dùng cho production?

**A:** Phụ thuộc vào mục tiêu:
- Nếu muốn **giảm tối đa chi phí LLM**: dùng `full`
- Nếu muốn **cân bằng**: dùng `hl_only`
- Nếu muốn **kết quả tốt nhất** (dựa trên test): dùng `none` (baseline LLM) hoặc `full` (xem kết quả RAGAS thực tế)

### Q: Có thể chạy evaluation mà không có LLM API key không?

**A:** Phần A (frame extraction nội bộ) và keyword extraction evaluation cho mode `hl_only` / `full` **không cần API key**. Phần B (RAGAS scoring) cần OpenAI API key vì RAGAS dùng LLM để tính metrics.

### Q: `frame-semantic-transformer` có hỗ trợ tiếng Việt không?

**A:** Không. Model được train trên FrameNet tiếng Anh. Với corpus tiếng Việt, mode `full` sẽ cho kết quả kém. Khuyến nghị:
- Dùng mode `none` (LLM) cho tiếng Việt
- Hoặc dùng `hl_only` với LLM cho ll_keywords

### Q: Tại sao mode `full` vẫn cần LLM?

**A:** LLM vẫn được dùng cho **de-duplicate và summarise** — gộp và tóm tắt mô tả của các thực thể/quan hệ xuất hiện nhiều lần. Đây là bước quan trọng để đảm bảo chất lượng đồ thị tri thức. Việc thay thế bước này bằng frame-semantic nằm ngoài phạm vi dự án hiện tại.

### Q: Làm sao kiểm tra chế độ đang dùng?

```python
from lightrag.operate import FRAME_EXTRACTION_MODE
print(FRAME_EXTRACTION_MODE)  # "none", "hl_only", hoặc "full"
```

Hoặc xem log khi indexing:
```
[full] Chunk 1/5 → 3 Ent + 2 Rel  chunk-abc123
[hl_only] Chunk 1/5 → 5 Ent + 8 Rel  chunk-abc123
```

### Q: Lỗi `ModuleNotFoundError: No module named 'frame_semantic_transformer'`?

```bash
# Cài đặt extras
pip install -e ".[frame-semantic]"

# Hoặc thủ công
pip install "frame-semantic-transformer>=0.1.0,<1.0.0" "protobuf>=3.20.1,<4.0.0"
```

### Q: Lỗi xung đột protobuf khi cài cùng google-genai?

Tạo môi trường Conda riêng theo hướng dẫn mục 8. Không có cách nào cài cả hai trong cùng một môi trường pip.

---

*Tài liệu này mô tả trạng thái codebase tính đến ngày 2026-05-18.*
*Tác giả thay đổi: Claude Sonnet 4.6 theo yêu cầu của người dùng.*
