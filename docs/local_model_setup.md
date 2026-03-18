# Local Model Setup Guide — M4 Pro 48GB + RTX 5090 32GB (March 2026)
# Using `llama.cpp` native (not Ollama)

## Why `llama.cpp` Native

- First-class Apple Metal and CUDA support
- OpenAI-compatible API server via `llama-server`
- Full control over quantization, context length, sampling, and chat-template kwargs
- Correct handling of reasoning-capable models such as Qwen3.5
- Easier to debug than another abstraction layer on top

---

## Step 0: Build `llama.cpp`

Build once per machine. The same commands work whether you later run local files or use direct Hugging Face loading with `-hf`.

### Apple Silicon / M4 Pro 48GB

Metal is enabled by default on modern macOS builds of `llama.cpp`, so you do **not** need a special Metal flag unless upstream changes again.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build --config Release -j "$(sysctl -n hw.ncpu)" \
  --target llama-cli llama-server llama-mtmd-cli

# quick sanity check
./build/bin/llama-bench --help

# downloader
python3 -m pip install -U huggingface_hub
```

### RTX 5090 32GB

For Blackwell, if you set CUDA arch explicitly, use compute capability `12.0`, i.e. `CMAKE_CUDA_ARCHITECTURES=120`.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j \
  --target llama-cli llama-server llama-mtmd-cli

# quick sanity check
./build/bin/llama-bench --help

# downloader
python3 -m pip install -U huggingface_hub
```

---

## Step 1: Verified model lineup, repos, and files

The original draft mixed `Qwen3.5-27B` with `Qwen3-32B`. Those are not the same model family, so this guide now uses the currently verified Qwen3.5 lineup consistently.

| Model | Official base model | Verified GGUF repo | M4 Pro default | RTX 5090 default | Optional higher-quality 5090 pick | Extra file |
|-------|---------------------|--------------------|----------------|------------------|-----------------------------------|-----------|
| **Qwen3.5-27B** | `Qwen/Qwen3.5-27B` | `unsloth/Qwen3.5-27B-GGUF` | `Qwen3.5-27B-UD-Q4_K_XL.gguf` | `Qwen3.5-27B-UD-Q4_K_XL.gguf` | `Qwen3.5-27B-Q5_K_M.gguf` | `mmproj-F16.gguf` |
| **Qwen3.5-35B-A3B** | `Qwen/Qwen3.5-35B-A3B` | `unsloth/Qwen3.5-35B-A3B-GGUF` | `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` | `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` | `Qwen3.5-35B-A3B-Q5_K_M.gguf` if it fits your workload | `mmproj-F16.gguf` |
| **Gemma 3 27B IT** | `google/gemma-3-27b-it` | `unsloth/gemma-3-27b-it-GGUF` | `gemma-3-27b-it-Q4_K_M.gguf` | `gemma-3-27b-it-Q4_K_M.gguf` | `gemma-3-27b-it-Q5_K_M.gguf` | `mmproj-F16.gguf` |
| **Phi-4** | `microsoft/phi-4` | `MaziyarPanahi/phi-4-GGUF` | `phi-4.Q5_K_M.gguf` | `phi-4.Q5_K_M.gguf` | `phi-4.Q6_K.gguf` | none |

### Notes on the lineup

- **Qwen3.5-27B**: multimodal, native context `262,144`, thinking enabled by default
- **Qwen3.5-35B-A3B**: multimodal MoE, `35B` total / `3B` active per token, native context `262,144`
- **Gemma 3 27B IT**: multimodal, `128K` input context, `8192` output context
- **Phi-4**: dense `14B`, text-only, `16K` context

### Official Gemma alternative

If you want Google's official GGUF rather than the Unsloth K-quants, use:

- repo: `google/gemma-3-27b-it-qat-q4_0-gguf`
- model: `gemma-3-27b-it-q4_0.gguf`
- projector: `mmproj-model-f16-27B.gguf`

For local benchmarking, the Unsloth `Q4_K_M` / `Q5_K_M` files are usually the more familiar choice.

---

## Step 2: Download the models

For text-only ALP work, the multimodal projector files are optional at runtime, but it is still convenient to download them now so the local model folder is complete.

```bash
mkdir -p ./models/qwen3.5-27b ./models/qwen3.5-35b-a3b ./models/gemma3-27b ./models/phi4

# 1. Qwen3.5-27B
# M4 Pro / safe default:
huggingface-cli download unsloth/Qwen3.5-27B-GGUF \
  --include "Qwen3.5-27B-UD-Q4_K_XL.gguf" \
  --include "mmproj-F16.gguf" \
  --local-dir ./models/qwen3.5-27b
# RTX 5090 higher-quality option: replace include with "Qwen3.5-27B-Q5_K_M.gguf"

# 2. Qwen3.5-35B-A3B
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  --include "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf" \
  --include "mmproj-F16.gguf" \
  --local-dir ./models/qwen3.5-35b-a3b
# Speed-first alternative: use "Qwen3.5-35B-A3B-MXFP4_MOE.gguf"

# 3. Gemma 3 27B IT (Unsloth K-quant)
# M4 Pro / safe default:
huggingface-cli download unsloth/gemma-3-27b-it-GGUF \
  --include "gemma-3-27b-it-Q4_K_M.gguf" \
  --include "mmproj-F16.gguf" \
  --local-dir ./models/gemma3-27b
# RTX 5090 higher-quality option: replace include with "gemma-3-27b-it-Q5_K_M.gguf"

# 4. Phi-4
# M4 Pro / safe default:
huggingface-cli download MaziyarPanahi/phi-4-GGUF \
  --include "phi-4.Q5_K_M.gguf" \
  --local-dir ./models/phi4
# RTX 5090 higher-quality option: replace include with "phi-4.Q6_K.gguf"
```

### Optional: official Gemma 3 GGUF instead of Unsloth

```bash
huggingface-cli download google/gemma-3-27b-it-qat-q4_0-gguf \
  --include "gemma-3-27b-it-q4_0.gguf" \
  --include "mmproj-model-f16-27B.gguf" \
  --local-dir ./models/gemma3-27b-official
```

---

## Step 3: Launch models with `llama-server`

These launch recipes are for **text-only** ALP / STS benchmarking, so projector files are omitted. If you later want image input on Qwen3.5 or Gemma 3, add `--mmproj /path/to/mmproj-F16.gguf` and use the multimodal entrypoints as needed.

### Common benchmark flags

- `-ngl 99` — aggressively offload layers to GPU/Metal
- `-fa` — flash attention
- `-c 8192` — 8K context, enough for this benchmark design
- `-n 4096` — generous max generation limit
- `--temp 0.0 --top-k 1` — deterministic, reproducible runs

### Qwen3.5-27B — thinking mode

```bash
./build/bin/llama-server \
  --model ./models/qwen3.5-27b/Qwen3.5-27B-UD-Q4_K_XL.gguf \
  --reasoning-format deepseek \
  -ngl 99 -fa -c 8192 -n 4096 \
  --temp 0.0 --top-k 1 \
  --host 0.0.0.0 --port 8080
```

### Qwen3.5-27B — non-thinking mode

This is the correct replacement for the old `/no_think` approach.

```bash
./build/bin/llama-server \
  --model ./models/qwen3.5-27b/Qwen3.5-27B-UD-Q4_K_XL.gguf \
  --reasoning-format deepseek \
  --chat-template-kwargs '{"enable_thinking":false}' \
  -ngl 99 -fa -c 8192 -n 4096 \
  --temp 0.0 --top-k 1 \
  --host 0.0.0.0 --port 8080
```

### Qwen3.5-35B-A3B — thinking mode

```bash
./build/bin/llama-server \
  --model ./models/qwen3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --reasoning-format deepseek \
  -ngl 99 -fa -c 8192 -n 4096 \
  --temp 0.0 --top-k 1 \
  --host 0.0.0.0 --port 8080
```

### Qwen3.5-35B-A3B — non-thinking mode

```bash
./build/bin/llama-server \
  --model ./models/qwen3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --reasoning-format deepseek \
  --chat-template-kwargs '{"enable_thinking":false}' \
  -ngl 99 -fa -c 8192 -n 4096 \
  --temp 0.0 --top-k 1 \
  --host 0.0.0.0 --port 8080
```

### Gemma 3 27B IT

```bash
./build/bin/llama-server \
  --model ./models/gemma3-27b/gemma-3-27b-it-Q4_K_M.gguf \
  -ngl 99 -fa -c 8192 -n 4096 \
  --temp 0.0 --top-k 1 \
  --host 0.0.0.0 --port 8080
```

### Phi-4

```bash
./build/bin/llama-server \
  --model ./models/phi4/phi-4.Q5_K_M.gguf \
  -ngl 99 -fa -c 8192 -n 4096 \
  --temp 0.0 --top-k 1 \
  --host 0.0.0.0 --port 8080
```

### Optional: direct HF loading instead of local paths

If you built with `-DLLAMA_CURL=ON`, you can also load directly from Hugging Face:

```bash
./build/bin/llama-server \
  -hf unsloth/Qwen3.5-27B-GGUF:UD-Q4_K_XL \
  --reasoning-format deepseek \
  -ngl 99 -fa -c 8192 -n 4096 \
  --temp 0.0 --top-k 1 \
  --host 0.0.0.0 --port 8080
```

---

## Step 4: Hyperparameters — benchmark mode vs interactive mode

These are two different goals and should stay separated.

### For ALP / STS benchmarking

Use deterministic settings:

- `temperature = 0.0`
- `top_k = 1`
- fixed `max_tokens`
- fixed `ctx_size`

This gives cleaner reproducibility across runs and models.

### For interactive sanity checks

Use the family-specific defaults below.

#### Qwen3.5 thinking mode

General tasks:

- `temperature = 1.0`
- `top_p = 0.95`
- `top_k = 20`
- `min_p = 0.0`
- `presence_penalty = 1.5`
- `repetition_penalty = 1.0`

Precise coding tasks:

- `temperature = 0.6`
- `top_p = 0.95`
- `top_k = 20`
- `min_p = 0.0`
- `presence_penalty = 0.0`
- `repetition_penalty = 1.0`

#### Qwen3.5 non-thinking mode

Disable reasoning with:

- `--chat-template-kwargs '{"enable_thinking":false}'`

General tasks:

- `temperature = 0.7`
- `top_p = 0.8`
- `top_k = 20`
- `min_p = 0.0`
- `presence_penalty = 1.5`
- `repetition_penalty = 1.0`

Reasoning tasks without explicit chain-of-thought mode:

- `temperature = 1.0`
- `top_p = 0.95`
- `top_k = 20`
- `min_p = 0.0`

#### Gemma 3 official recommended inference settings

- `temperature = 1.0`
- `top_k = 64`
- `top_p = 0.95`
- `min_p = 0.0` to `0.01`
- `repetition_penalty = 1.0`

Gemma 3 note: `llama.cpp` already adds a BOS token, so do **not** manually prepend a second `<bos>`.

#### Phi-4

Phi-4's official model card clearly specifies chat formatting, but does **not** publish a distinctive official decoding recipe like Qwen3.5 or Gemma 3. For local exploratory use, standard conservative settings are fine. For this benchmark, keep the deterministic setup above.

---

## Step 5: Python wrapper (OpenAI-compatible API)

`llama-server` exposes an OpenAI-compatible API at `http://localhost:8080/v1`, so the normal `openai` client works directly.

```python
from openai import OpenAI
import re


client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")


class LocalLLM:
    """Drop-in replacement for a simple local chat model."""

    def __init__(self, model_name: str = "local"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="http://localhost:8080/v1",
            api_key="not-needed",
        )

    def prompt(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": text}],
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""


class MultiTurnSession:
    def __init__(self, model_name: str = "local"):
        self.model_name = model_name
        self.messages: list[dict] = []
        self.client = OpenAI(
            base_url="http://localhost:8080/v1",
            api_key="not-needed",
        )

    def send(self, text: str, role: str = "user") -> str:
        self.messages.append({"role": role, "content": text})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            max_tokens=512,
            temperature=0.0,
        )
        reply = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def inject_system(self, text: str):
        self.messages.insert(0, {"role": "system", "content": text})

    def reset(self):
        self.messages = []


def extract_answer(response: str) -> str:
    """Strip Qwen reasoning blocks and return the final answer text."""
    text = response.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    match = re.search(r"Output:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[-1] if lines else text


def extract_thinking(response: str) -> str | None:
    """Extract inline Qwen-style <think> blocks when present."""
    match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
    return match.group(1).strip() if match else None
```

---

## Step 6: Running Qwen3.5 in both thinking and non-thinking modes

This is still one of the most interesting parts of the setup, but the toggle has to be done correctly.

- **Thinking mode**: default Qwen3.5 behavior
- **Non-thinking mode**: launch server with `--chat-template-kwargs '{"enable_thinking":false}'`

Do **not** try to disable Qwen3.5 reasoning by stuffing `/no_think` into the system prompt. That was the main behavioral bug in the original draft.

This effectively gives you an extra experimental condition for the same weights:

- `Qwen3.5-27B-thinking`
- `Qwen3.5-27B-non-thinking`
- and the same split if you want to do it with `Qwen3.5-35B-A3B`

---

## Step 7: Hardware fit guidance

No made-up throughput table here — measure your own token/s once the boxes are actually running. The practical fit guidance is:

| Hardware | Recommended default set | Notes |
|---------|--------------------------|-------|
| **M4 Pro 48GB unified** | Qwen3.5-27B `UD-Q4_K_XL`, Qwen3.5-35B-A3B `UD-Q4_K_XL`, Gemma 3 `Q4_K_M`, Phi-4 `Q5_K_M` | All should fit one-at-a-time comfortably for 8K benchmark contexts |
| **RTX 5090 32GB** | Same defaults as M4 Pro | You can often step up to Q5 on Qwen3.5-27B, Gemma 3 27B, and Phi-4 if VRAM headroom stays healthy |

If a run goes out of memory:

1. reduce context length first
2. then reduce max generation length
3. only then step down the quant

---

## Step 8: Quick validation test

Run this before you trust any benchmark output.

```bash
./build/bin/llama-server \
  --model ./models/qwen3.5-27b/Qwen3.5-27B-UD-Q4_K_XL.gguf \
  --reasoning-format deepseek \
  -ngl 99 -fa -c 4096 \
  --temp 0.0 --top-k 1 \
  --port 8080 &

sleep 10

curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Below are examples of a symbolic transformation system.\n\n◈⬡ → ⟐\n⬡◈ → ⧫\n◈◈ → ⬡\n\nInput: ◈⬡◈\nOutput:"}],
    "max_tokens": 64,
    "temperature": 0
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

If you get a symbol sequence back instead of an API or template error, the pipeline is alive.

---

## Step 9: What to log from reasoning models

For Qwen3.5, the `<think>...</think>` traces are useful qualitative data. Keep them, but evaluate final answers separately.

```python
strategy_keywords = {
    "rule_mention": [r"rule", r"pattern", r"always", r"whenever", r"if.*then"],
    "exemplar_mention": [r"similar to", r"like example", r"same as", r"looks like"],
    "uncertainty": [r"not sure", r"unclear", r"might be", r"guess"],
    "self_correction": [r"wait", r"actually", r"no,", r"let me reconsider"],
}
```

That gives you a clean bridge from quantitative metrics to interpretable strategy evidence.
