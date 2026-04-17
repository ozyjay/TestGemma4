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

## Clean clone setup

On a new Windows machine:

```powershell
git clone <your-repo-url>
cd TestGemma4

py -3.12 -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
huggingface-cli login
```

The Hugging Face account must have accepted the Gemma model license before the
first model download.

Run from source:

```powershell
python app.py
```

On first launch, the app asks where to create `.test.gemma4`. That folder stores
the saved Assistant Behaviour (`system_prompt.md`), chat logs, and diagnostics
logs. The model weights are not stored in this repo; Hugging Face downloads and
caches them on the machine.

The desktop app also saves the current conversation in:

```text
.test.gemma4\conversation.json
```

When you reopen the app, the previous conversation is restored. Use
`Clear Conversation` to start fresh. This resets `conversation.json` to an empty
conversation but does not delete transcript or diagnostics log files.

If Git reports dubious ownership after cloning or moving the folder, trust the
clone path for the current Windows user:

```powershell
git config --global --add safe.directory <full-path-to-clone>
```

Example:

```powershell
git config --global --add safe.directory C:/Users/You/source/TestGemma4
```

Build and install the desktop app:

```powershell
.\scripts\Build.ps1 -Clean
.\scripts\Install-ToPrograms.ps1 -Replace
```

Default install path:

```text
%LOCALAPPDATA%\Programs\Gemma4Chat
```

Run the installed app:

```powershell
& "$env:LOCALAPPDATA\Programs\Gemma4Chat\Gemma4Chat.exe"
```

Custom install location:

```powershell
.\scripts\Install-ToPrograms.ps1 -InstallRoot "D:\Programs" -Replace
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
The rewrite uses thinking mode internally; any thinking output appears in the
Thinking pane.

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

Install the built app under the current user's local programs folder:

```powershell
.\scripts\Install-ToPrograms.ps1
```

By default this installs to:

```text
%LOCALAPPDATA%\Programs\Gemma4Chat
```

Earlier versions of this script hardcoded `E:\Programs` as the default install
root. The script is now computer-agnostic, so `E:\Programs` is only used when you
ask for it explicitly.

Choose a custom install location when needed:

```powershell
.\scripts\Install-ToPrograms.ps1 -InstallRoot "E:\Programs"
```

That installs to:

```text
E:\Programs\Gemma4Chat
```

Replace an existing installed copy:

```powershell
.\scripts\Install-ToPrograms.ps1 -Replace
```

Replace an existing copy in a custom location:

```powershell
.\scripts\Install-ToPrograms.ps1 -InstallRoot "E:\Programs" -Replace
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
installed under your local programs folder or another explicit `-InstallRoot`.

## Diagnostics

Use the `Diagnostics` toolbar button to show or hide captured stdout/stderr output.
Diagnostics are captured even while the pane is hidden and are saved beside the
chat logs:

```text
.test.gemma4\diagnostics_YYYYMMDD_HHMMSS.log
```
