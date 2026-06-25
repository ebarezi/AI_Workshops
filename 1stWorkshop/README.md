# Workshop 1: LoRA Fine-Tuning

Fine-tune a small language model using **LoRA** (Low-Rank Adaptation) on the Alpaca instruction dataset.  
Only ~0.2% of parameters are trained while the base model stays frozen.

## Contents

- `Workshop1_LoRA_Finetuning.ipynb` — interactive teaching notebook. All dependencies are installed from its own **Section 0** cell (`%pip install ...`) — no separate environment file is required.

## Setup

### Local machine (conda)

```bash
conda create -n workshop1 python=3.12 -y
conda activate workshop1
pip install ipykernel
python -m ipykernel install --user --name workshop1 --display-name "Python (workshop1)"
```

Open the notebook, select the `Python (workshop1)` kernel, then run Section 0's `%pip install` cell once and restart the kernel.

### Purdue Gilbreth cluster

Before launching this notebook, create and select a conda environment kernel (Workshop1) from a terminal on the cluster:

```bash
module load conda
conda-env-mod create -n ENV_NAME_HERE -j
module use $HOME/privatemodules
module load conda-env/ENV_NAME_HERE-py3.12.11
```

Replace `ENV_NAME_HERE` with your environment name (Workshop1), then select the matching kernel in Jupyter before running the cells below.

## Running the notebook

Open `Workshop1_LoRA_Finetuning.ipynb` in VS Code or Jupyter and select the `Python (workshop1)` kernel.

Or launch from the terminal:

```bash
jupyter lab Workshop1_LoRA_Finetuning.ipynb
```

## What the notebook does

| Section | Description |
|---|---|
| 0 | Install dependencies (`%pip install` cell — run once, then restart the kernel) |
| 1–2 | Imports and model selection (`model_name`, default `Qwen/Qwen2.5-0.5B-Instruct`) |
| 3 | Prompt formatting using Qwen's native **ChatML** format (`build_messages` + `apply_chat_template`) |
| 4 | Load and split `yahma/alpaca-cleaned` (600 samples, 80/10/10) or your own local JSON/CSV |
| 5 | Tokenize — loss computed on the full sequence (prompt + response) |
| 5b | **Alternative:** response-only tokenization — prompt tokens masked to `-100` so loss is computed on the assistant turn only |
| 6 | Load base model with 4-bit quantization (QLoRA) when a GPU is available, otherwise full-precision fp32 on CPU; gradient checkpointing enabled on the GPU path via `prepare_model_for_kbit_training` |
| 7 | Attach LoRA adapter (`r=8`, `lora_alpha=32`, `target_modules=["q_proj","k_proj","v_proj","o_proj"]`) |
| 8 | `MultiEvalTrainer` (optional) — evaluates both validation and test sets at every checkpoint |
| 9 | Training arguments — GPU-only settings (`fp16`, gradient checkpointing, paged optimizer) bundled into a `gpu_settings` dict; early stopping (patience=3) |
| 10 | Train |
| 11 | Plot train / validation / test loss curves |
| 12 | Save the LoRA adapter (~5 MB) — the tokenizer is *not* re-saved, since LoRA training leaves it unmodified |
| 13 | Reload the adapter onto a freshly loaded base model and run inference using the ChatML template |
| 14 | Extension ideas |

### Tokenization: Section 5 vs 5b

By default **Section 5** runs (no masking — loss on the full sequence).  
To train with response-only loss, run **Section 5b instead** — it sets prompt token labels to `-100` so the model only learns to predict the assistant response.

## Notes

- Change `model_name` in Section 2 to swap models without touching anything else.
- On CPU (or Apple Silicon) the notebook automatically skips 4-bit quantization and runs in fp32 — slower but functional; the default `Qwen/Qwen2.5-0.5B-Instruct` finishes the default 30-step run in a few minutes.
- For GPU training, an A10-class GPU or better is recommended for the 4-bit QLoRA path.

## Troubleshooting

- If `bitsandbytes` fails to install or import, confirm you have CUDA-compatible drivers and Python 3.10–3.12 — it's only required for the GPU/QLoRA path; CPU and Apple Silicon runs don't need it to succeed.
- If the notebook fails due to missing packages, verify the active kernel matches the environment where you ran Section 0's installs.
- A gradient checkpointing warning is safe to ignore at this model size; Section 6 has a commented-out `gradient_checkpointing_kwargs={"use_reentrant": False}` you can uncomment to silence it.
