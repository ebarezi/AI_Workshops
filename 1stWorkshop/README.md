# Workshop 1: LoRA Fine-Tuning

Fine-tune a small language model using **LoRA** (Low-Rank Adaptation) on the Alpaca instruction dataset.  
Only ~0.3% of parameters are trained while the base model stays frozen.

## Contents

- `Workshop1_LoRA_Finetuning.ipynb` — interactive teaching notebook
- `environment.yml` — Conda/Mamba environment specification (recommended)
- `requirements.txt` — pip dependency list

## Setup

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate 1stWorkshop
python -m ipykernel install --user --name 1stWorkshop --display-name "Python (1stWorkshop)"
```

### Option B: pip

```bash
python3 -m pip install -r requirements.txt
```

## Running the notebook

Open `Workshop1_LoRA_Finetuning.ipynb` in VS Code or Jupyter and select the `Python (1stWorkshop)` kernel.

Or launch from the terminal:

```bash
jupyter lab Workshop1_LoRA_Finetuning.ipynb
```

## What the notebook does

| Section | Description |
|---|---|
| 1–2 | Imports and model selection |
| 3 | Prompt formatting using Qwen's native **ChatML** format (`build_messages` + `apply_chat_template`) |
| 4 | Load and split `yahma/alpaca-cleaned` (600 samples, 80/10/10) or your own local JSON/CSV |
| 5 | Tokenize — loss computed on the full sequence (prompt + response) |
| 5b | **Alternative:** response-only tokenization — prompt tokens masked to `-100` so loss is computed on the assistant turn only |
| 6 | Load base model with 4-bit quantization (QLoRA); gradient checkpointing enabled with `use_reentrant=False` |
| 7 | Attach LoRA adapter (`r=8`, `lora_alpha=32`, attention projections) |
| 8 | `MultiEvalTrainer` — evaluates both validation and test sets at every checkpoint |
| 9 | Training arguments + early stopping (patience=3) |
| 10 | Train |
| 11 | Plot train / validation / test loss curves |
| 12 | Save the LoRA adapter (~5 MB) |
| 13 | Reload adapter and run inference using the ChatML template |
| 14 | Extension ideas |

### Tokenization: Section 5 vs 5b

By default **Section 5** runs (no masking — loss on the full sequence).  
To train with response-only loss, run **Section 5b instead** — it sets prompt token labels to `-100` so the model only learns to predict the assistant response.

## Notes

- Change `model_name` in Section 2 to swap models without touching anything else.
- On CPU the notebook is slow but functional; the default `Qwen/Qwen2.5-0.5B-Instruct` is small enough to complete.
- For GPU training, an A10-class GPU or better is recommended for the 4-bit QLoRA path.

## Troubleshooting

- If `bitsandbytes` fails, confirm you have CUDA-compatible drivers and Python 3.10–3.12.
- If the notebook fails due to missing packages, verify the active kernel matches the environment where you installed dependencies.
- If you see a gradient checkpointing warning, ensure you are using PEFT ≥ 0.7 and Transformers ≥ 4.36 (both pinned in `environment.yml`).
