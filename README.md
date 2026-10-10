# Tree-CoT-RT cho Aspect Sentiment Quad Prediction

Project triển khai flow **Tree-CoT-RT: An Explainable Multi-Path Tree-Guided Chain-of-Thought and Reinforcement Learning Framework for Aspect Sentiment Quad Prediction** theo paper [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.806/).

Input của bạn ở `asqp_data/` có `train.jsonl`, `dev.jsonl`, `test.jsonl`. Mỗi dòng phải có `instance_id`, `text`, `quads`; các trường khác được bỏ qua. Mỗi quad dùng đúng bốn trường:

- `aspect_term`: chuỗi hoặc `null`
- `aspect_category`: nhãn category
- `opinion_term`: chuỗi hoặc `null`
- `sentiment`: nhãn sentiment

Loader giữ thông tin implicit từ `aspect_implicit`/`opinion_implicit` khi có, và cũng coi term `null` là implicit. Một câu có thể chứa nhiều quad thuộc các loại explicit/implicit khác nhau; project nhóm chúng theo bốn loại của paper: **EE, EA, IE, IA** (aspect trước, opinion sau) để huấn luyện và suy luận riêng từng reasoning tree.

## Pipeline paper

1. **S1-a — Multi-path Tree-CoT generation:** tạo `K=10` reasoning chains cho từng cặp câu/loại quad bằng các prompt tree riêng cho EE/EA/IE/IA. Các chain lưu vào `stage1_candidates.jsonl`.
2. **S1-b — tSFT:** lọc chain parse được và hợp lệ theo loại tree, rồi LoRA fine-tune Qwen2.5-7B-Instruct (rank 8, alpha 16, dropout 0.05, 15 steps) để khởi tạo policy.
3. **S2 — GRPO-RL và offline selection:** GRPO group size 10, clipping 0.2, KL 0.01; reward gộp consistency với gold, validity của JSON/quad type/ordered tree steps, cue cho implicit reasoning, và diversity so với các chain cùng câu/type (mặc định mỗi thành phần 0.25). Lưu policy và các chain chọn ở `stage2_grpo_policy/` và `stage2_selected_chains.jsonl`.
4. **S3 — reasoning-aware SFT:** khởi tạo model cuối Qwen2.5-7B, fine-tune LoRA trên chain được chọn cùng gold quadruples; adapter nằm trong `final/`.
5. **S4 — self-consistent inference:** lấy mẫu nhiều chain riêng cho từng loại tree, bỏ nhãn không đúng loại, sau đó majority vote quadruple trong từng nhóm để không làm mất các loại quad hiếm.

Các giá trị mặc định bám bảng hyperparameter paper; model size và dữ liệu bên dưới khác cấu hình benchmark gốc. Cần GPU Kaggle đủ bộ nhớ (khuyến nghị A100 40GB trở lên) và Internet để tải model/dependencies. Việc sinh chain và GRPO trên 8,815 training records có thể tốn nhiều giờ/GPU; output S1/S2 được cache thành JSONL để chạy tiếp mà không tạo lại chain đã có.

## Điểm khác với thiết lập paper

- Paper đánh giá trên Restaurant-ACOS và Laptop-ACOS; project dùng dataset khách sạn được cung cấp, 8,815 train / 1,895 dev / 1,890 test và sáu category (`FACILITY`, `AMENITY`, `SERVICE`, `EXPERIENCE`, `LOYALTY`, `BRANDING`). Các F1 không thể so sánh trực tiếp với bảng kết quả của paper.
- Paper tạo S1 chains qua API ChatGPT/Gemini. Để notebook Kaggle chạy độc lập với API key, implementation này dùng `Qwen2.5-7B-Instruct` local làm chain generator. Đây là thay đổi có chủ ý; nếu thay generator thì các chain và kết quả có thể khác paper.
- Mặc định chạy một seed `444`; paper báo trung bình bốn seed `444/555/666/777` trên bốn A100.
- Repo GitHub được dẫn trong paper hiện không có source code công khai; đây là implementation độc lập theo mô tả paper, không phải bản sao official.

## Chạy local

Tại thư mục `tree/`:

```powershell
python -m pip install -e ".[dev]"
tree-cot-rt train `
  --train asqp_data/train.jsonl `
  --dev asqp_data/dev.jsonl `
  --model Qwen/Qwen2.5-7B-Instruct `
  --final-model Qwen/Qwen2.5-7B `
  --output-dir runs/tree_cot_rt
```

Inference trên toàn bộ test set với năm sample cho mỗi tree:

```powershell
tree-cot-rt predict `
  --input asqp_data/test.jsonl `
  --adapter-dir runs/tree_cot_rt/final `
  --base-model Qwen/Qwen2.5-7B `
  --samples 5 `
  --output runs/tree_cot_rt/test_predictions.jsonl

tree-cot-rt evaluate `
  --gold asqp_data/test.jsonl `
  --predictions runs/tree_cot_rt/test_predictions.jsonl
```

Metric là exact-match micro-precision/recall/F1: một quad chỉ đúng khi đủ cả bốn trường khớp trong đúng `instance_id`; không chấm partial match.

Các tham số paper có thể giảm cho smoke test, ví dụ `--chains-per-type 1 --grpo-steps 1 --selected-chains-per-type 1 --final-steps 1 --max-new-tokens 256`. Kết quả smoke test không đại diện kết quả nghiên cứu.

## Chạy trên Kaggle

1. Upload thư mục project (bao gồm `pyproject.toml` và `src/tree_cot_rt/`) thành Kaggle Dataset.
2. Upload `asqp_data/` thành một Kaggle Dataset khác hoặc gom chung vào source dataset.
3. Tạo notebook mới từ `tree_cot_rt_kaggle.ipynb`, bấm **Add Input** cho hai dataset, bật **GPU** và **Internet** trong Notebook settings.
4. Sửa `PROJECT_DIR`/`DATA_DIR` ở cell cấu hình nếu auto-detect không chọn đúng input.
5. Run all; adapters, candidate chains, prediction JSONL và metric được ghi trong `/kaggle/working/tree_cot_rt_outputs/`.

Notebook có chế độ smoke test qua các biến `CHAINS_PER_TYPE`, `GRPO_STEPS`, `FINAL_STEPS`, `MAX_TRAIN_RECORDS`, `MAX_TEST_RECORDS`. Để chạy đầy đủ, để `MAX_*_RECORDS = None` và giữ các giá trị hyperparameter mặc định.

## Kiểm thử

```powershell
python -m pytest
```
# tree
