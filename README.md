# V-ABS: Action-Observer Driven Beam Search for Dynamic Visual Reasoning

<p align="center">
  <a href="https://github.com/pami-zwning/V-ABS"><img src="https://img.shields.io/badge/GitHub-V--ABS-blue?logo=github" alt="GitHub"></a>
  <a href="https://arxiv.org/abs/2605.10172"><img src="https://img.shields.io/badge/arXiv-Paper-red" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/ICML-2026-green" alt="ICML 2026">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</p>

Official implementation of the ICML 2026 paper:

> **V-ABS: Action-Observer Driven Beam Search for Dynamic Visual Reasoning**
> Accepted at ICML 2026

---

## Overview

Multimodal large language models (MLLMs) often suffer from **Imagination-Action-Observer (IAO) bias** — a critical misalignment between a model's prior imagination of an action's outcome and the actual observed result after execution. This leads to suboptimal search trajectories in agentic visual reasoning tasks.

**V-ABS** addresses this problem through a unified framework that integrates three tightly-coupled modules into a beam search loop:

- **Thinker**: Evaluates candidate actions by computing prior scores from the model's predicted log-probabilities over positive tokens (`Yes/True/Correct`), without executing the action.
- **Actor**: Executes the selected candidate actions via external tool functions (e.g., `Image.Crop`, `Image.Rearrange`), transitioning the visual state.
- **Observer**: Queries the model on the *updated* visual state to produce grounded posterior scores, providing feedback to correct the Thinker's prior bias.

An **entropy-based adaptive weighting mechanism** dynamically balances the Thinker and Observer scores: when the Thinker is uncertain (high entropy), the Observer's signal dominates, and vice versa. A complementary **SFT dataset (~80k samples)** fine-tunes the model to reduce intrinsic prior uncertainty and improve thinker calibration.

### Key Results

| Task | Baseline | V-ABS | Gain |
|------|---------|-------|------|
| V* (Overall) | 75.9% | **90.5%** | +14.6% |
| HR-Bench 4K | 70.0% | **76.0%** | +6.0% |
| VisuoThink (Lv3-5) | 22.4% | **46.1%** | +23.7% |
| Frozen Lake | 51.5% | **78.8%** | +27.3% |
| TIR-Bench Maze | 36.7% | **56.7%** | +20.0% |
| Jigsaw | 57.8% | **81.2%** | +23.4% |

*Results on Qwen3-VL-8B backbone. Average improvement of +19.7% across all 8 benchmarks.*

---

## Repository Structure

```
V-ABS/
├── config.py                     # API and model endpoint configuration
├── utils_llm.py                  # Unified VLM call interface (Qwen, InternVL, GPT-4o, Gemini)
├── utils.py                      # General utility functions (I/O, threading, logging)
│
├── unified_visual_search/        # Visual Search task module (V*, HR-Bench)
│   ├── agent.py                  # VisualSearchAgent: hierarchical beam search
│   ├── tool.py                   # Image.Crop tool implementation
│   ├── prompt.py                 # Prompt templates for thinker/observer/answer
│   └── run.py                    # Evaluation entry point
│
├── unified_navigation/           # Visual Navigation task module (VisuoThink, Frozen Lake, TIR-Bench)
│   ├── agent.py                  # NavigationAgent: sequential path planning beam search
│   ├── env.py                    # Grid environment simulator
│   ├── prompt.py                 # Navigation-specific prompts
│   └── run.py                    # Evaluation entry point
│
├── unified_jigsaw/               # Visual Manipulation task module (Jigsaw, Sudoku)
│   ├── agent.py                  # JigsawAgent: permutation beam search
│   ├── env.py                    # Jigsaw/Sudoku environment
│   ├── prompt.py                 # Manipulation-specific prompts
│   └── run.py                    # Evaluation entry point
│
└── scripts/                      # Dataset generation and utility scripts
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/pami-zwning/V-ABS.git
cd V-ABS

# Install dependencies
pip install openai requests pillow tqdm jsonlines
```

### Optional Dependencies

- **vLLM** (for local open-source models): Follow [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- **openai-proxy** (for GPT-4o via proxy): Install your organization's proxy client

---

## Configuration

Before running, configure your model endpoints in `config.py`:

```python
# For GPT-4o
GPT_PROXY_URL = "YOUR_OPENAI_PROXY_URL"  # e.g., "https://api.openai.com"
GPT_KEY = "YOUR_OPENAI_API_KEY"

# For Qwen/InternVL via vLLM
QWEN_URL_DICT = {
    "qwen_8b_instruct": ["http://<your-vllm-server>:8000/v1"],
}
INTERNVL_URL_DICT = {
    "internvl3_8b": ["http://<your-vllm-server>:8000/v1"],
}

# For Gemini
GEMINI_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_URL = "YOUR_GEMINI_API_URL"
```

---

## Datasets

Download the benchmark datasets and place them under `datasets/`:

| Dataset | Link | Path |
|---------|------|------|
| V* Benchmark | [GitHub](https://github.com/penghao-wu/vstar) | `datasets/visual-search/vstar/` |
| HR-Bench 4K/8K | [HuggingFace](https://huggingface.co/datasets/DreamMr/HR-Bench) | `datasets/visual-search/hr_bench_4k/` |
| VisuoThink | [GitHub](https://github.com/nianticlabs/visuothink) | `datasets/navigation/visuothink/` |
| Frozen Lake | Synthesized (see `scripts/`) | `datasets/navigation/frozen_lake/` |
| TIR-Bench Maze | [GitHub](https://github.com/TIR-Bench) | `datasets/navigation/tir_bench/` |
| Jigsaw | Based on RefCOCO + `scripts/` | `datasets/jigsaw/` |
| Sudoku | [AlgoPuzzleVQA](https://github.com/declare-lab/LLM-PuzzleTest) | `datasets/sudoku/` |

---

## Usage

### Visual Search (V*, HR-Bench)

```bash
# V-ABS beam search on V* benchmark
python unified_visual_search/run.py \
    --dataset_type vstar \
    --type beam \
    --model_name qwen_8b_instruct \
    --beam_width 3 \
    --max_depth 4 \
    --num_threads 8

# Direct VQA baseline (no search)
python unified_visual_search/run.py \
    --dataset_type hr_bench_4k \
    --type direct \
    --model_name qwen_8b_instruct
```

### Visual Navigation

```bash
python unified_navigation/run.py \
    --dataset_type visuothink \
    --type beam \
    --model_name qwen_8b_instruct \
    --beam_width 3 \
    --max_depth 4
```

### Visual Manipulation (Jigsaw / Sudoku)

```bash
python unified_jigsaw/run.py \
    --dataset_type jigsaw \
    --type beam \
    --model_name qwen_8b_instruct \
    --beam_width 3 \
    --max_depth 3
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `qwen_8b_instruct` | VLM backend (see `config.py` for available options) |
| `--beam_width` | `3` | Beam size K (recommended: 2–5) |
| `--max_depth` | `4` | Maximum search depth D |
| `--num_threads` | `8` | Parallel threads for concurrent VLM queries |
| `--num_workers` | `1` | Parallel processes for dataset-level acceleration |
| `--type` | `beam` | `beam` = V-ABS search; `direct` = direct VQA baseline |
| `--entropy_skip_threshold` | `-1.0` | Skip observer when prior entropy < δ (disabled if -1.0) |
| `--mu_threshold` | `0.5` | Entropy threshold μ for adaptive weighting |

### Background Execution

```bash
nohup python -u unified_visual_search/run.py \
    --dataset_type vstar \
    --type beam \
    --model_name qwen_8b_instruct \
    > run_vstar.log 2>&1 &
tail -f run_vstar.log
```

---

## Evaluation

After running, results are saved to `outputs/{model_name}/visual_search_{dataset}/predictions_{type}_bw{K}_d{D}.jsonl`.

To compute accuracy:

```bash
python eval.sh  # or use the eval script for the specific task
```

---

## SFT Training

To fine-tune a model using our SFT dataset (~80k binary verification samples):

```bash
# Dataset generation scripts are in scripts/
# SFT training scripts (LLaMA-Factory / swift compatible):
python scripts/generate_sft_data.py --task visual_search
```

The SFT dataset covers three domains:
- **Visual Search**: 7,040 crop action verification samples
- **Visual Navigation**: 14,000 directional action verification samples  
- **Visual Logic (Jigsaw)**: 60,000 permutation verification samples (3×3, 4×4, 5×5)

---

## Algorithm

V-ABS formalizes visual reasoning as a Markov Decision Process and applies beam search with thinker-actor-observer closed-loop feedback:

```
For each step t = 0 ... D-1:
  Thinker: compute prior scores F_pri over all candidate actions (parallel)
  w_p = sigmoid(-β * (H_t - μ))   [entropy-based adaptive weight]
  
  For each action b:
    Actor:    execute action → update visual state v_t+1
    Observer: compute posterior score F_obs on updated state
    Heuristic: compute task-specific score F_heur
    Score: S = w_p * F_pri + w_o * F_obs + F_heur
  
  Keep top-K states → new beam B_t+1
  Terminate if best node satisfies stopping condition
```

The adaptive weight `w_p = 1 / (1 + exp(β*(H_t - μ)))` ensures that:
- **High entropy** (uncertain prior) → rely more on Observer (`w_o → 1`)
- **Low entropy** (confident prior) → trust the Thinker (`w_p → 1`)
- When entropy < δ (optional acceleration), skip the Observer entirely

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{vabs2026,
  title     = {V-ABS: Action-Observer Driven Beam Search for Dynamic Visual Reasoning},
  author    = {Anonymous},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://github.com/pami-zwning/V-ABS}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

We thank the authors of [V* Benchmark](https://github.com/penghao-wu/vstar), [HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench), [VisuoThink](https://github.com/nianticlabs/visuothink), [TIR-Bench](https://github.com/TIR-Bench), and [AlgoPuzzleVQA](https://github.com/declare-lab/LLM-PuzzleTest) for providing their datasets and evaluation protocols.
