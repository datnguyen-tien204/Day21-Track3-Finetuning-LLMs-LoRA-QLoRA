# Lab 21 — LoRA / QLoRA Fine-tuning Report

**Họ tên:** Nguyễn Tiến Đạt

**MSV:** 2A202600217

**Repo Hugging Face:** https://huggingface.co/datnguyentien204/qwen2.5-3b-vi-lab21-r16  

**Colab:** https://colab.research.google.com/drive/1TiflPHeDuzQkhbzXGcidHSs7Ddihj-GV?usp=sharing  

**Notebook:** `notebooks\Lab21_LoRA_Finetuning_T4_FullBonus_Complete.ipynb`  

**Submission option:** Option B — Hugging Face Hub, kèm các bonus artifacts: `r16_all_layers`, GGUF export, multi-tenant adapter swap demo.

**Extend.1 Flash Attention:** Flash Attention usage| RTX 4060Ti| L40s: `notebooks\lab21-flashattn2-qwen35-4b-l40s.ipynb`

**Extend.2 Flash Attention:** Flash Attention usage| RTX 4060Ti| L40s: `notebooks\lab21-flexattn-qwen35-4b-l40s.ipynb`

---

## 1. Setup

### 1.1 Base model

Trong bài lab này, em chọn base model:

```text
unsloth/Qwen2.5-3B-bnb-4bit
```

Lý do chọn model này là Qwen2.5-3B có kích thước phù hợp với Google Colab T4 16GB, hỗ trợ tốt tác vụ tiếng Việt / multilingual, và bản `bnb-4bit` của Unsloth giúp giảm VRAM khi fine-tune bằng QLoRA. Mô hình được load với `load_in_4bit=True`, sau đó gắn LoRA adapter để chỉ train một phần nhỏ tham số thay vì cập nhật toàn bộ model.

### 1.2 Dataset

Dataset được sử dụng:

```text
5CD-AI/Vietnamese-alpaca-gpt4-gg-translated
```

Notebook lấy 200 samples đầu tiên và dùng các cột tiếng Việt:

```text
instruction_vi, input_vi, output_vi
```

Dữ liệu được đưa về Alpaca format:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

Token length analysis cho 200 samples:

| Metric | Value |
|---|---:|
| min | 25 |
| p50 | 227 |
| p95 | 562 |
| p99 | 704 |
| max | 738 |

Vì p95 = 562, notebook chọn:

```text
max_seq_length = 1024
```

Đây là lựa chọn hợp lý vì 1024 bao phủ phần lớn samples, đồng thời vẫn an toàn cho T4 khi train QLoRA.

Split dữ liệu:

| Split | Số lượng |
|---|---:|
| Train | 180 |
| Eval | 20 |

### 1.3 GPU / environment

Notebook chạy trên Google Colab với GPU:

| Thành phần | Giá trị |
|---|---|
| GPU | Tesla T4 |
| VRAM | 15.6 GB |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Framework chính | Unsloth, TRL SFTTrainer, PEFT, bitsandbytes |

### 1.4 LoRA / QLoRA configuration

Baseline dùng LoRA rank r=16 theo yêu cầu lab:

```python
r = 16
lora_alpha = 32
target_modules = ["q_proj", "v_proj"]
lora_dropout = 0
gradient_checkpointing = True
```

Training hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Epochs | 3 |
| Per-device train batch size | 1 |
| Gradient accumulation steps | 8 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.10 |
| Optimizer | adamw_8bit |
| Eval split | 10% |

### 1.5 Training cost estimate

Tổng thời gian train core experiment r=8, r=16, r=64 là khoảng **12.4 phút**. Nếu ước tính chi phí T4 là **0.35 USD/giờ**, chi phí train core experiment xấp xỉ:

```text
12.4 / 60 * 0.35 ≈ 0.07 USD
```

---

## 2. Rank Experiment Results

Bài lab train 3 adapter LoRA với cùng base model, cùng dataset, cùng hyperparameters; chỉ thay đổi `rank` và `lora_alpha`:

| Rank | Alpha | Trainable params | Train time (min) | Peak VRAM (GB) | Eval loss | Eval perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 4.13 | 7.22 | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 4.22 | 6.62 | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 4.04 | 8.00 | 1.4768 | 4.3790 |

### Nhận xét định lượng

Kết quả cho thấy khi tăng rank từ 8 lên 16 rồi 64, eval perplexity giảm lần lượt từ **4.7479 → 4.5544 → 4.3790**. Điều này nghĩa là adapter rank cao hơn học được biểu diễn tốt hơn trên eval set. Tuy nhiên, số lượng trainable parameters tăng rất nhanh: r=64 có **14.75M** trainable parameters, gấp **4 lần r=16** và **8 lần r=8**. Mức cải thiện perplexity từ r=16 sang r=64 là khoảng **0.175**, nhỏ hơn so với chi phí tham số tăng thêm.

Một điểm thú vị là thời gian train giữa các rank khá gần nhau trên T4, dao động khoảng 4.0–4.2 phút. Điều này có thể do dataset chỉ có 180 training samples nên overhead load model, tokenize, logging và Colab runtime chiếm tỷ trọng lớn. Tuy vậy, VRAM peak vẫn phản ánh rõ hơn chi phí của rank cao: r=64 dùng khoảng **8.00GB**, cao hơn r=16 khoảng **1.38GB**.

### Base model perplexity

Notebook có qualitative comparison giữa base model và r=16 fine-tuned model, nhưng **chưa log eval perplexity riêng cho base model**. Vì vậy, báo cáo không tự suy đoán con số base perplexity. Nếu cần hoàn thiện tuyệt đối theo rubric, nên chạy thêm một cell evaluate base model trên cùng `eval_ds` để có đủ 4 số: base, r=8, r=16, r=64.

---

## 3. Loss Curve Analysis

Trong T4 mode, notebook tắt eval-during-training để tiết kiệm VRAM và tránh lỗi OOM, nên loss curve chính là **training loss**. Với baseline r=16, log training loss giảm ở giai đoạn đầu rồi dao động nhẹ ở giai đoạn cuối. Một số điểm loss cuối của r=16 nằm quanh **1.38–1.42**, cho thấy model tiếp tục học nhưng không giảm đều tuyệt đối.

Vì không có eval loss theo từng checkpoint trong lúc train, chưa thể khẳng định chắc chắn có overfitting theo tiêu chí “train loss giảm nhưng eval loss tăng”. Tuy nhiên, có một vài dấu hiệu cần lưu ý:

1. Dataset chỉ có **200 samples**, trong đó train set chỉ **180 samples**, nên nguy cơ overfitting là có thật.
2. Rank cao như r=64 có nhiều tham số trainable hơn đáng kể, do đó có khả năng fit nhanh hơn vào training set.
3. Eval perplexity của r=64 là tốt nhất trong 3 rank, nên trong experiment hiện tại chưa thấy bằng chứng định lượng rằng r=64 bị overfit. Tuy vậy, do eval set chỉ có 20 samples, kết luận này chưa đủ mạnh.

Kết luận phần loss curve: quá trình train chạy ổn định, không có dấu hiệu diverge. Chưa có bằng chứng rõ ràng về overfitting, nhưng cần eval loss theo checkpoint và eval set lớn hơn để kết luận chắc chắn hơn.

---

## 4. Qualitative Comparison

Notebook tạo qualitative comparison giữa base model và adapter r=16 trên 20 prompts. Dưới đây là 5 examples tiêu biểu, gồm cả case tốt và case chưa tốt.

| # | Prompt | Base model | Fine-tuned r=16 | Nhận xét |
|---:|---|---|---|---|
| 1 | Giải thích khái niệm machine learning cho người mới bắt đầu. | Trả lời đúng hướng, nhưng hơi dài và lặp ý. | Trả lời cũng đúng hướng, dễ hiểu hơn ở phần mở đầu, nhưng vẫn bị cắt ở cuối. | r=16 cải thiện nhẹ về phrasing tiếng Việt, nhưng chưa hoàn toàn ổn định. |
| 2 | Viết đoạn code Python tính số Fibonacci thứ n. | Có code lặp, nhưng xử lý chỉ số hơi thiếu nhất quán. | Code rõ hơn, có `ValueError` cho input âm, xử lý `n=0`, `n=1`. | Đây là case r=16 tốt hơn base model. |
| 3 | Liệt kê 5 nguyên tắc thiết kế UI/UX. | Trả lời đúng chủ đề nhưng dài, hơi lan man. | Trả lời dạng list 5 ý ngắn hơn: chuyển đổi, thích ứng, đơn giản, tương thích, nội dung. | r=16 bám format list tốt hơn, dù một số ý chưa thật chuẩn thuật ngữ. |
| 4 | Tóm tắt sự khác biệt giữa LoRA và QLoRA. | Có giải thích LoRA/QLoRA nhưng còn mơ hồ. | Có lỗi nghiêm trọng: gọi LoRA là “Layer-wise Adaptive Regularization Optimization”. | Đây là case r=16 thua base; model hallucinate thuật ngữ. |
| 5 | Giải thích overfitting và cách phòng tránh. | Base giải thích đúng khái niệm overfitting: train tốt nhưng generalize kém. | Fine-tuned trả lời lệch chủ đề, nói về “bài báo trên báo chí”. | Đây là failure case, cho thấy fine-tuning với dataset nhỏ có thể làm model kém ổn định ở một số prompt. |

### Nhận xét qualitative

Fine-tuned r=16 không thắng tuyệt đối base model. Adapter r=16 cải thiện một số prompt yêu cầu format rõ ràng, đặc biệt là code Python và list ngắn. Tuy nhiên, ở một số prompt kỹ thuật sâu như LoRA/QLoRA hoặc overfitting, model vẫn hallucinate hoặc trả lời lệch. Điều này có thể đến từ ba nguyên nhân: dataset nhỏ, domain của Vietnamese Alpaca khá general, và model chỉ train 3 epochs với target modules mặc định `q_proj`, `v_proj`. Vì vậy, kết quả qualitative nên được đọc cùng với quantitative perplexity: perplexity giảm cho thấy model fit eval set tốt hơn, nhưng chưa đảm bảo mọi câu trả lời đều chính xác về mặt chuyên môn.

---

## 5. Conclusion về Rank Trade-off

Trong experiment này, rank cao hơn giúp cải thiện eval perplexity: r=64 đạt perplexity thấp nhất (**4.3790**), r=16 đứng thứ hai (**4.5544**), và r=8 kém nhất (**4.7479**). Điều này phù hợp với trực giác về LoRA: rank càng cao thì adapter càng có nhiều capacity để biểu diễn update cho weight matrix của base model. Tuy nhiên, capacity tăng không miễn phí. r=64 có **14.75M** trainable parameters, gấp 4 lần r=16, nhưng perplexity chỉ cải thiện thêm khoảng **0.175** so với r=16. Với dataset chỉ 200 samples, mức tăng tham số này có thể không phải lựa chọn tối ưu nếu mục tiêu là cân bằng giữa chất lượng, VRAM và kích thước adapter.

Theo em, **r=16 là lựa chọn tốt nhất cho submission chính**. r=16 có perplexity tốt hơn r=8 rõ ràng, dùng ít VRAM hơn r=64, adapter nhỏ hơn, dễ upload và deploy hơn. r=64 phù hợp nếu mục tiêu là tối ưu điểm perplexity và có đủ tài nguyên, nhưng ROI không cao bằng r=16 trong bối cảnh dataset nhỏ và GPU T4. r=8 là lựa chọn tiết kiệm nhất, nhưng chất lượng thấp hơn và có thể underfit. Vì vậy, r=16 là điểm cân bằng hợp lý giữa hiệu quả học, chi phí tính toán và tính thực tế khi triển khai.

---

## 6. What I learned

- Em hiểu rõ hơn trade-off của LoRA rank: rank thấp tiết kiệm tham số nhưng dễ thiếu capacity, rank cao cải thiện perplexity nhưng tăng VRAM và kích thước adapter.
- QLoRA 4-bit kết hợp Unsloth giúp fine-tune model 3B trên Tesla T4 khả thi, với chi phí rất thấp và thời gian train ngắn.
- Perplexity thấp hơn không đảm bảo model luôn trả lời đúng. Cần kết hợp quantitative evaluation với qualitative prompts, đặc biệt phải kiểm tra các failure cases như hallucination thuật ngữ hoặc trả lời lệch chủ đề.

---

## Bonus Experiments

### Bonus A — ALL layers vs q+v only

Ngoài baseline `target_modules=["q_proj", "v_proj"]`, notebook train thêm một cấu hình r=16 target ALL layers:

```python
target_modules = [
  "q_proj", "k_proj", "v_proj", "o_proj",
  "gate_proj", "up_proj", "down_proj"
]
```

Kết quả:

| Config | Rank | Trainable params | Train time (min) | Peak VRAM (GB) | Eval perplexity |
|---|---:|---:|---:|---:|---:|
| q+v only | 16 | 3,686,400 | 4.2 | 6.6 | 4.554 |
| ALL layers | 16 | 29,933,568 | 4.9 | 10.6 | 4.459 |

ALL layers cải thiện perplexity khoảng **0.096**, nhưng trainable params tăng hơn **8 lần** và VRAM tăng khoảng **4GB**. Đây là một improvement có thật, nhưng chi phí khá lớn. Với T4, cấu hình ALL layers vẫn chạy được, nhưng không phải lựa chọn tiết kiệm nhất.

### Bonus C — GGUF export

Notebook export adapter r=16 sang GGUF `q4_k_m`, giúp deploy model bằng llama.cpp / Ollama / LM Studio. Repo Hugging Face có GGUF artifact và hướng dẫn dùng với `llama.cpp`, `llama-cpp-python`, Ollama, Docker Model Runner.

### Bonus D — Multi-tenant adapter swap

Notebook demo pattern dùng một base model và swap nhiều adapter r=8, r=16, r=64. Kết quả load adapter khoảng dưới 1 giây:

| Adapter | Load time | VRAM | Generation time |
|---|---:|---:|---:|
| r8 | 0.22s | 11.0GB | 8.7s |
| r16 | 0.24s | 11.0GB | 7.4s |
| r64 | 0.83s | 11.1GB | 7.7s |

Pattern này hữu ích trong production vì chỉ cần giữ một base model trong memory, còn mỗi tenant/domain có thể dùng một adapter nhỏ riêng.

---

## Additional Extension — Qwen3.5-4B Attention Experiments

Phần này là **bổ sung sau submission chính**, không thay thế kết quả chính ở trên. Submission chính vẫn là:

```text
Base model chính: unsloth/Qwen2.5-3B-bnb-4bit
Fine-tuning chính: QLoRA 4-bit trên Google Colab Tesla T4 15.6GB
Repo chính: datnguyentien204/qwen2.5-3b-vi-lab21-r16
```

Hai notebook bổ sung được viết để thử một hướng mạnh hơn: thay base model Qwen2.5-3B 4-bit bằng **Qwen3.5-4B bf16 LoRA**, đồng thời thay backend attention để kiểm tra khả năng train nhanh hơn / tối ưu hơn. Vì vậy, phần này nên được hiểu là **extension experiment**: giữ lại toàn bộ pipeline LoRA/rank experiment của bài chính, nhưng thay model, precision, attention backend, và cách target LoRA layers.

> Ghi chú: tên file upload có chữ `L40s`, nhưng nội dung notebook đang ghi GPU target là **RTX 4060 Ti 16GB, Ada Lovelace, SM 8.9**. Báo cáo này mô tả theo nội dung notebook. Nếu thực tế chạy trên L40S thì chỉ cần đổi dòng GPU thành **NVIDIA L40S 48GB**; các ý chính về bf16 LoRA, FA2/FlexAttention và ALL-layer LoRA vẫn giữ nguyên.

### Extension summary — 2-column comparison

| Hạng mục | FlashAttention-2 Extension | FlexAttention Extension |
|---|---|---|
| Notebook | `Lab21_FLASHATTN2_Qwen35_4B_L40s.ipynb` | `Lab21_FLEXATTN_Qwen35_4B_L40si.ipynb` |
| Vai trò trong báo cáo | Bản mở rộng dùng FA2 để tối ưu attention kernel | Bản mở rộng dùng FlexAttention built-in của PyTorch |
| Có thay submission chính không? | Không. Submission chính vẫn là Qwen2.5-3B QLoRA trên T4 | Không. Đây là bản so sánh thêm với backend attention khác |
| Thay model gì? | Thay `unsloth/Qwen2.5-3B-bnb-4bit` bằng `Qwen/Qwen3.5-4B` | Tương tự: thay sang `Qwen/Qwen3.5-4B` |
| Thay precision / quantization gì? | Không dùng QLoRA 4-bit nữa; chuyển sang **bf16 LoRA** với `load_in_4bit=False`, `load_in_16bit=True` | Tương tự: **bf16 LoRA**, không dùng 4-bit |
| Lý do bỏ 4-bit | Notebook ghi rõ Qwen3.5 không khuyến nghị QLoRA 4-bit; dùng bf16 LoRA để ổn định hơn | Tương tự, ưu tiên bf16 để tránh rủi ro chất lượng / compatibility |
| GPU target trong notebook | RTX 4060 Ti 16GB, Ada Lovelace, CUDA SM 8.9; cần SM >= 8.0 cho FA2/bf16 | RTX 4060 Ti 16GB, Ada Lovelace; FlexAttention dùng PyTorch built-in |
| Attention backend | `flash_attention_2` qua package `flash-attn` và Unsloth kernels | `flex_attention`, không cần cài riêng `flash-attn` |
| Cài đặt thêm | Có cài `flash-attn --no-build-isolation` | Không cài `flash-attn`; dùng PyTorch >= 2.5 built-in FlexAttention |
| Dataset | Vẫn dùng `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` | Tương tự |
| Số mẫu | Tăng từ 200 samples ở bài chính lên **300 samples** | Tương tự |
| Split | 90/10 train/eval, tức khoảng 270 train và 30 eval | Tương tự |
| Max sequence length | `max_seq_length = 1024`, chọn theo phân tích token length p95 và cap 1024 | Tương tự |
| LoRA target modules | Chuyển từ q+v baseline sang **ALL layers**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Tương tự |
| LoRA rank chính | Baseline r=16, sau đó train thêm r=8 và r=64 | Baseline r=16, sau đó train thêm r=8 và r=64 |
| LoRA alpha | r=8 alpha=16, r=16 alpha=32, r=64 alpha=128 | Tương tự |
| Batch strategy | Train batch size 1, gradient accumulation 8, effective batch size 8 | Tương tự |
| Optimizer | `adamw_torch_fused`, bf16 enabled | `adamw_torch_fused`, bf16 enabled |
| Eval strategy | Eval theo epoch vì GPU target đủ VRAM hơn T4 | Tương tự |
| Gradient checkpointing | `use_gradient_checkpointing="unsloth"` để giảm VRAM | Tương tự |
| Packing | `packing=False` để tránh lỗi dimension mismatch | Tương tự |
| Output directory | `~/lab21_lora_flashattn2` | `~/lab21_lora_flexattn` |
| Artifacts tạo ra | r8/r16/r64 adapters, merged model, GGUF, rank summary, qualitative CSV, plots | r8/r16/r64 adapters, merged model, GGUF, rank summary, qualitative CSV, plots |
| Bonus thêm | Benchmark FA2 vs PyTorch SDPA bằng `scaled_dot_product_attention` và `flash_attn_func` | So sánh hướng dùng FlexAttention không phụ thuộc package FA2 |
| Mục tiêu chính | Kiểm tra hiệu quả khi dùng FA2 kernel trên GPU Ada Lovelace | Kiểm tra pipeline tương đương nhưng dùng attention backend built-in |

### Cụ thể đã bổ sung cái gì?

Hai notebook mới bổ sung một hướng thí nghiệm lớn hơn so với báo cáo chính. Ở bản chính, em fine-tune **Qwen2.5-3B bản 4-bit** bằng QLoRA trên T4. Ở phần mở rộng, em thử dùng **Qwen3.5-4B** để kiểm tra xem pipeline có mở rộng được sang model lớn hơn và mới hơn hay không. Do Qwen3.5 trong notebook không dùng 4-bit, phần mở rộng chuyển sang **bf16 LoRA**. Đây là thay đổi quan trọng nhất: bản chính tối ưu để vừa T4 bằng 4-bit, còn bản mở rộng ưu tiên độ ổn định của model bằng bf16.

Thay đổi thứ hai là attention backend. Notebook FA2 dùng **FlashAttention-2**, cần GPU hỗ trợ SM >= 8.0 và cần cài package `flash-attn`. Notebook FlexAttention dùng **FlexAttention** của PyTorch, không cần package `flash-attn` riêng. Mục đích là so sánh hai cách tối ưu attention: một cách dựa trên thư viện FA2 phổ biến, một cách dựa trên backend mới hơn tích hợp trong PyTorch. Cả hai vẫn giữ cùng logic LoRA để so sánh công bằng: cùng dataset, cùng số mẫu, cùng split, cùng rank experiment, cùng max sequence length và cùng training hyperparameters chính.

Thay đổi thứ ba là target modules của LoRA. Ở bài chính, baseline dùng `q_proj` và `v_proj`, sau đó có bonus ALL layers. Trong hai notebook extension, ALL layers trở thành cấu hình mặc định. Cụ thể LoRA được gắn vào toàn bộ attention projection và FFN projection:

```python
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

Cách này làm adapter có nhiều trainable parameters hơn, nhưng thường cho khả năng học tốt hơn vì model không chỉ điều chỉnh query/value trong attention mà còn điều chỉnh cả output projection và MLP/FFN layers. Nói cách khác, phần mở rộng không chỉ thay backend attention, mà còn dùng cấu hình LoRA mạnh hơn so với baseline q+v của bài chính.

### Đã xử lý pipeline như thế nào?

Pipeline xử lý của hai notebook extension vẫn đi theo cấu trúc bài Lab21, nhưng được nâng cấp ở một số điểm:

1. **Environment check:** notebook kiểm tra GPU bằng `torch.cuda.get_device_name`, VRAM, compute capability, CUDA và PyTorch version. Với bản FA2, notebook assert GPU phải có compute capability từ SM 8.0 trở lên vì FA2/bf16 cần GPU đời mới. Notebook ghi target là RTX 4060 Ti 16GB, SM 8.9.

2. **Dataset preparation:** vẫn dùng Vietnamese Alpaca GPT-4 translated dataset, nhưng tăng số mẫu từ 200 lên 300. Dataset được shuffle với seed 42, tự động detect các cột `instruction_vi`, `input_vi`, `output_vi`, rồi format về Alpaca prompt template. Sau đó notebook tính token length distribution, dùng p95 để chọn `max_seq_length`, nhưng cap ở 1024 để giữ an toàn VRAM.

3. **Model loading:** thay vì load base 4-bit như bài chính, notebook load `Qwen/Qwen3.5-4B` với `dtype=torch.bfloat16`, `load_in_4bit=False`, `load_in_16bit=True`. Đây là phần “thay thế” rõ nhất: bỏ QLoRA 4-bit, dùng bf16 LoRA.

4. **LoRA wrapping:** model được wrap bằng Unsloth `FastLanguageModel.get_peft_model`, dùng rank r=16 cho baseline, alpha=32, dropout=0, bias none, gradient checkpointing Unsloth. Target mặc định là ALL layers.

5. **Training:** dùng TRL `SFTTrainer`/`SFTConfig`, batch size 1, gradient accumulation 8, learning rate 2e-4, cosine scheduler, warmup steps 10, eval theo epoch, save theo epoch, load best model at end theo eval loss. So với bản T4 chính, extension bật eval theo epoch vì GPU target có nhiều headroom hơn.

6. **Rank experiment:** sau baseline r=16, notebook train thêm r=8 và r=64, sau đó lưu bảng `rank_experiment_summary.csv` gồm train time, peak VRAM, eval loss và eval perplexity. Cấu trúc này giữ đúng tinh thần rubric: phải phân tích trade-off giữa rank thấp, rank trung bình và rank cao.

7. **Qualitative evaluation:** notebook chạy 20 prompts thay vì chỉ 5 prompts. Mỗi prompt có output của base model và output của fine-tuned r=16 adapter để so sánh before/after. Kết quả được lưu vào `qualitative_comparison_20.csv`.

8. **Deployment artifacts:** extension vẫn có bước merge adapter vào base model, export GGUF q4_k_m để chạy bằng llama.cpp/Ollama, và demo multi-tenant adapter swap. Điều này chứng minh pipeline không chỉ train được mà còn có hướng deploy.

### GPU dùng trong phần bổ sung

Theo nội dung notebook, GPU target là:

```text
RTX 4060 Ti 16GB
Ada Lovelace
CUDA SM 8.9
bf16 native support
```

Lý do chọn GPU này cho extension là vì Qwen3.5-4B bf16 LoRA cần nhiều VRAM hơn so với Qwen2.5-3B 4-bit trên T4. Bản chính dùng T4 15.6GB và phải tối ưu bằng 4-bit QLoRA. Bản extension dùng bf16 LoRA nên cần GPU hỗ trợ bf16 tốt và attention kernel hiện đại. RTX 4060 Ti 16GB đáp ứng được điều kiện này, đặc biệt với FA2 vì FA2 yêu cầu GPU SM >= 8.0.

Nếu chạy trên **NVIDIA L40S 48GB** như tên file gợi ý, phần này còn thuận lợi hơn: L40S cũng là GPU Ada Lovelace, hỗ trợ bf16, hỗ trợ FA2, và có VRAM lớn hơn nhiều so với 4060 Ti/T4. Khi đó có thể tăng batch size, tăng số samples, tăng max sequence length hoặc bật thêm evaluation/generation nhiều hơn mà ít rủi ro OOM hơn.

### Vì sao phần này không thay thế kết quả chính?

Phần extension có cấu hình mạnh hơn, nhưng không nên dùng để thay thế báo cáo chính vì nó thay đổi quá nhiều biến cùng lúc:

- Thay base model từ Qwen2.5-3B sang Qwen3.5-4B.
- Thay QLoRA 4-bit sang bf16 LoRA.
- Thay target modules từ q+v baseline sang ALL layers.
- Tăng dataset từ 200 lên 300 samples.
- Thay GPU/runtime từ Colab T4 sang GPU Ada Lovelace.
- Thay attention backend sang FA2 hoặc FlexAttention.

Vì nhiều biến thay đổi cùng lúc, kết quả extension không còn là so sánh trực tiếp với bản chính. Nó phù hợp để trình bày như một phần “em thử mở rộng thêm”, chứng minh rằng pipeline Lab21 có thể scale lên model mới hơn và backend attention tối ưu hơn. Còn kết luận chính về rank trade-off trong submission vẫn nên dựa vào bảng Qwen2.5-3B QLoRA trên T4 ở các phần trước.

### Kết luận phần extension

Hai notebook FA2 và FlexAttention cho thấy em đã mở rộng bài lab theo hướng production hơn: dùng model mới hơn, precision bf16 ổn định hơn, target ALL layers, rank experiment đầy đủ, qualitative comparison 20 prompts, GGUF export và multi-tenant adapter swap. Bản FA2 tập trung vào tối ưu kernel attention bằng FlashAttention-2, còn bản FlexAttention tập trung vào hướng dùng attention backend built-in của PyTorch. Cả hai đều giữ cấu trúc bài chính nhưng thay thế các thành phần quan trọng để kiểm tra khả năng scale của pipeline.

Tóm lại, phần chính chứng minh em hoàn thành yêu cầu lab trên T4 với QLoRA; phần bổ sung chứng minh em hiểu cách nâng cấp pipeline khi có GPU mạnh hơn và muốn dùng model/attention backend hiện đại hơn.

---

## Reproducibility Notes

Các artifacts chính trong repo Hugging Face:

```text
adapter/
r8/
r16/
r64/
r16_all_layers/
r16_gguf/
r16_merged_fp16/
results/
rank_experiment_summary.csv
qualitative_comparison.csv
qualitative_comparison_20.csv
bonus_a_ablation.csv
bonus_d_multitenant.csv
loss_r16.png
rank_comparison.png
token_dist.png
```

Lệnh load adapter r=16:

```python
from peft import PeftModel
from unsloth import FastLanguageModel

base, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-bnb-4bit",
    max_seq_length=1024,
    load_in_4bit=True,
)

model = PeftModel.from_pretrained(
    base,
    "datnguyentien204/qwen2.5-3b-vi-lab21-r16/adapter"
)
```

Lệnh chạy GGUF bằng llama.cpp / Ollama có thể xem trực tiếp trên Hugging Face model page.
