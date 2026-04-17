# TestGemma4

Test Google's **Gemma 4 E2B-it** model locally on an RTX 3060 (12 GB VRAM).

## Prerequisites

- Python **3.12** (PyTorch does not support 3.14 yet)
- NVIDIA GPU with CUDA drivers installed
- A [Hugging Face](https://huggingface.co/) account — you must accept the
  [Gemma 4 license](https://huggingface.co/google/gemma-4-E2B-it) before downloading

## Setup

```powershell
# Create a venv with Python 3.12 (using py launcher)
py -3.12 -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Log in to Hugging Face (needed once for gated model access)
huggingface-cli login
```

## Usage

### Interactive chat

```powershell
python chat.py            # normal mode
python chat.py --think    # enable thinking / reasoning mode
```

Commands inside the chat session:
- `/think` — toggle thinking mode on/off
- `/reset` — clear conversation history
- `/behaviour <instruction>` — rewrite Assistant Behaviour from advice
- `/behavior <instruction>` — same as `/behaviour`
- `quit` — exit

In the desktop app, type a behaviour change in the input box and click
`Update Behaviour` to ask Gemma to rewrite the saved Assistant Behaviour
instructions without sending the advice as a normal chat message.

### Single-shot generation

```powershell
python generate.py "Explain quicksort in Python"
python generate.py --think "What is 25 * 37?"
python generate.py --max-tokens 512 --system "You are a poet." "Write a haiku about GPU computing"
```

## VRAM usage

The E2B-it model loads in BF16 and uses ~10 GB VRAM, leaving ~2 GB headroom on a 12 GB card.
Monitor with `nvidia-smi`.

## Build the desktop executable

Put your icon PNG at:

```text
assets\app-icon.png
```

Then build from PowerShell:

```powershell
.\scripts\Build.ps1
```

The script installs the build dependencies from `requirements-build.txt`, converts
`assets\app-icon.png` to `assets\app-icon.ico`, and runs PyInstaller with
`Gemma4Chat.spec`.

The executable is written to:

```text
dist\Gemma4Chat\Gemma4Chat.exe
```

Useful build options:

```powershell
.\scripts\Build.ps1 -Clean        # remove build/dist first
.\scripts\Build.ps1 -SkipInstall  # reuse already installed build deps
```

Install the built app to `E:\Programs\Gemma4Chat`:

```powershell
.\scripts\Install-ToPrograms.ps1
```

Replace an existing installed copy:

```powershell
.\scripts\Install-ToPrograms.ps1 -Replace
```

The executable does not bundle the Gemma model weights. On first launch it uses
the same Hugging Face cache/download flow as the Python app.

### Why the build is large

The PyInstaller build is expected to be large. Even without bundling Gemma model
weights, the app has to include a Python runtime plus ML dependencies such as
PyTorch, Transformers, Accelerate, Tokenizers, SentencePiece, and CUDA/PyTorch
support libraries.

Most of the size is in:

```text
dist\Gemma4Chat\_internal\
```

This project intentionally uses PyInstaller's one-folder layout:

```text
dist\Gemma4Chat\Gemma4Chat.exe
dist\Gemma4Chat\_internal\
```

Avoid `--onefile` for this app unless there is a specific reason. It would still
be large, and startup is usually slower because the bundled files have to be
unpacked before launch.

For daily use on your own machine, running from the virtual environment can be
lighter and easier to update:

```powershell
.\.venv\Scripts\python.exe app.py
```

Use the PyInstaller build when you want a self-contained app folder that can be
installed under `E:\Programs`.

## Diagnostics

Use the `Diagnostics` toolbar button to show or hide captured stdout/stderr output.
Diagnostics are captured even while the pane is hidden and are saved beside the
chat logs:

```text
.test.gemma4\diagnostics_YYYYMMDD_HHMMSS.log
```
