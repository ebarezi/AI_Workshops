# Workshop 1: LoRA Fine-Tuning

This folder contains the materials for the first Purdue workshop on LoRA fine-tuning.

## Contents

- `Workshop1_LoRA_Finetuning.ipynb` — interactive teaching notebook.
- `train_lora.py` — runnable script version of the same workflow.
- `outline.md` — workshop agenda and teaching plan.
- `slides.md` — slide notes for the instructor.
- `environment.yml` — Conda/Mamba environment specification.
- `requirements.txt` — pip dependency list.

## Setup

Create a fresh environment and install the workshop dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### (Optional) Create a Conda environment named `1stWorkshop`

To create a reproducible Conda environment and keep the workshop sandboxed, run:

```bash
conda env create -n 1stWorkshop python=3.10 
conda activate 1stWorkshop
pip install -r requirements.txt
```

Or, if you have an `environment.yml` you prefer to use:

```bash
conda env create -f environment.yml -n 1stWorkshop
conda activate 1stWorkshop
```

## Running the notebook

Launch Jupyter Lab or Notebook:

```bash
cd "$(dirname \"$(realpath 1stWorkshop/README.md)\")"
jupyter lab 1stWorkshop/Workshop1_LoRA_Finetuning.ipynb
```

or:

```bash
jupyter notebook 1stWorkshop/Workshop1_LoRA_Finetuning.ipynb
```

## Running the script

```bash
python 1stWorkshop/train_lora.py
```

## What the script does

`train_lora.py` runs the full fine-tuning workflow optimized for workshop speed:

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` — completes in minutes on a laptop, seconds on an A10. Change `model_name` at the top of the script to swap models.
- **Dataset:** 600 samples from `yahma/alpaca-cleaned`, or point it at a local JSON/CSV file (see the commented-out Option B block in the script).
- **Training:** 1 epoch, `max_length=512`, early stopping with patience=3 on validation loss.
- **Evaluation:** Validation and test loss are both evaluated every 5 steps, producing full loss curves.
- **Output:** Loss plot (train / validation / test) saved to `workshop_outputs/lora-alpaca/loss_curves.png`. Adapter saved to `lora_alpaca_adapter/`.

## Notes

- `environment.yml` is recommended for the most reproducible setup.
- `requirements.txt` is useful when using a plain `pip` virtual environment.
- If you run into package conflicts, create a fresh environment and install from one of the files only.
- `matplotlib` is required for the loss plot — it is included in `requirements.txt`.

## Troubleshooting

- If `torch` or `bitsandbytes` fails to install, ensure your Python version is 3.10–3.12 and that you have CUDA-compatible drivers installed.
- If the notebook fails due to missing packages, verify the active environment and install the required dependencies again.
- For GPU training, use a machine with at least one A10-class GPU and sufficient VRAM to load a 4-bit quantized model.
- On CPU-only machines the run is slow but functional — the default `Qwen2.5-0.5B-Instruct` model is small enough to complete in a reasonable time.
