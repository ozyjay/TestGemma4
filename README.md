# TestGemma4

Test Google's **Gemma 4 E2B-it** model locally.

This guide assumes you are using a **Windows 11** computer with an **NVIDIA GPU**.
The smoothest experience is on **12 GB+ VRAM**. An **8 GB VRAM** card can use the
project's automatic low-VRAM mode.

## Quick Windows setup

Gemma 4 is a local model, so setup depends on the computer's GPU, driver, Python
environment, and the ability to download from Hugging Face.

Before running the app for the first time:

1. Download or clone this project.
2. Open PowerShell in the project folder.
3. Check whether the computer looks suitable:

If Windows blocks local PowerShell scripts because of the execution policy, use
a temporary bypass for the current PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This only affects the current PowerShell process. Alternatively, run a script
with a one-shot bypass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Setup-Gemma4.ps1 -CheckOnly
```

```powershell
.\scripts\Setup-Gemma4.ps1 -CheckOnly
```

The setup assistant checks the GPU, NVIDIA driver, disk space, Python
environment, PyTorch CUDA support, and whether Hugging Face authentication is
available. Login is optional for public model downloads.

If the assistant prints `ACTION NEEDED`, follow the final `Next step` it shows,
then run the check again.

When the check looks suitable, run the setup:

```powershell
.\scripts\Setup-Gemma4.ps1
```

This creates `.venv` and installs the Python dependencies when needed.

You can pre-download the model:

```powershell
.\scripts\Setup-Gemma4.ps1 -PreDownloadModel
```

To launch the app after a successful setup:

```powershell
.\scripts\Setup-Gemma4.ps1 -Launch
```

The assistant cannot install NVIDIA drivers or install Python globally. It will
tell you when one of those manual steps is required.

## Hugging Face model location

The Gemma model files are large and are downloaded by Hugging Face into its
local cache. Expect the download/cache to need roughly **10-20 GB** for the
model files, plus extra free space for Python packages and temporary download
files. By default, the Hugging Face cache is usually under your Windows user
profile.

To store Hugging Face models on another drive, set `HF_HOME` before downloading
the model:

```powershell
[Environment]::SetEnvironmentVariable("HF_HOME", "E:\HuggingFace", "User")
```

Close and reopen PowerShell after setting it, then run:

```powershell
.\scripts\Setup-Gemma4.ps1 -PreDownloadModel
```

Use a folder on a drive with plenty of free space. The app will use the same
Hugging Face cache location when it starts.

## Manual developer setup

```powershell
# Create a venv with Python 3.12 (using py launcher)
py -3.12 -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: log in to Hugging Face if a download asks for authentication
hf auth login
```

Run from source:

```powershell
python app.py
```

On first launch, the app asks where to create `.test.gemma4`. That folder is the
first Behaviour Profile. A Behaviour Profile stores its own saved Assistant
Behaviour (`system_prompt.md`), prompt history, current conversation, chat logs,
and diagnostics logs. The model weights are not stored in this repo; Hugging
Face downloads and caches them on the machine.

Use the `Behaviour` toolbar button to show or hide the Assistant Behaviour
panel. Inside that panel, use the `Behaviour` selector to switch between
profiles. Use `New` to create a new profile folder seeded with the current
Assistant Behaviour, or `Add Existing` to attach an existing folder. Switching
profiles saves the current profile and restores the selected profile's prompt
and conversation.

Assistant Behaviour history is saved beside the current profile's prompt:

```text
.test.gemma4\system_prompt_history.json
```

Use the prompt history selector in the desktop app to restore an earlier
Assistant Behaviour. The Assistant Behaviour editor supports normal undo and
redo shortcuts while you are editing it.

The desktop app also saves the current profile's conversation in:

```text
.test.gemma4\conversation.json
```

When you reopen the app, the active profile's previous conversation is restored.
Use `Clear Conversation` to start fresh in the current profile. This resets that
profile's `conversation.json` to an empty conversation but does not delete
transcript or diagnostics log files.

Settings that remember the active and recent profiles are stored outside the
profile folders:

```text
%APPDATA%\TestGemma4\settings.json
```

Older installs that already used a single `.test.gemma4` folder are migrated by
treating that existing folder as the first active Behaviour Profile. No files
need to be moved manually.

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

In the desktop app, type a behaviour change in the input box and click
`Update Behaviour` to ask Gemma to rewrite the saved Assistant Behaviour
instructions without sending the advice as a normal chat message.
The rewrite uses thinking mode internally; any thinking output appears in the
Thinking pane.

The desktop app can keep multiple Behaviour Profiles. Each profile has its own
Assistant Behaviour, prompt history, restored conversation, transcripts, and
diagnostics. Use the `Behaviour` toolbar button to open the panel, then switch
profiles from the selector above the prompt editor. `New` creates a profile
from the current prompt, and `Add Existing` uses an existing profile folder.

The bottom stats bar also shows token usage after the model has loaded. The
token count includes the current Assistant Behaviour, restored conversation, and
the configured `Max tokens` reply budget. The indicator changes colour as usage
approaches the model context window; sending is disabled only when the prompt
itself no longer fits.

The token readout looks like this:

```text
Tokens: 6,900/8,192 (84%) - input 5,876 + reply budget 1,024
```

The `input` number is everything sent before the model starts replying. The
reply budget is the reserved output room from `Max tokens`. The token part of
the stats bar is normal below 80%, yellow/orange at 80% or higher, and red at
95% or higher. It also turns red if the input alone exceeds the model context
window. If the overall total is yellow or red but sending is still enabled,
reduce `Max tokens`, shorten Assistant Behaviour, or clear older conversation
context.

### Configuring Assistant Behaviour and knowledge

The Assistant Behaviour field is the app's system prompt. It is a good place for
stable instructions: tone, role, response style, formatting preferences,
decision rules, and small reference notes that should apply to every reply.

You can use it in a similar way to GPT Builder instructions by keeping two
sections in the prompt:

```text
## Instructions
- How the assistant should behave
- What it should prioritize
- How it should format answers

## Knowledge
- Compact facts, definitions, policies, examples, or project notes
- Only include material that is useful often enough to send every turn
```

The important difference is that this app does not currently retrieve snippets
from separate knowledge files on demand. Everything in Assistant Behaviour is
sent to the model on every message. Large knowledge packs therefore consume
context, reduce room for conversation, and can slow generation.

Use the bottom stats bar to tune the prompt. The `input` token count includes
Assistant Behaviour, restored conversation, the current user message, and chat
template overhead. The reply budget comes from `Max tokens`. A practical target
is to keep total usage below about 80% of the model context window for normal
chat, and treat 95% as the danger zone.

As a starting point:

- Keep core instructions around 500-2,000 tokens.
- Add compact knowledge only when it is needed across many turns.
- Keep the combined instructions plus always-on knowledge around 2,000-4,000
  tokens if the loaded model reports an 8,192 token context window.
- Lower `Max tokens` when you want more room for prompt or conversation.
- Use `Clear Conversation` when an old restored conversation is taking context
  away from a new task.

Behaviour Profiles are useful for different agent setups. For example, create
one profile for coding, another for writing, and another for a project-specific
assistant with its own compact knowledge section. Each profile keeps its own
Assistant Behaviour, history, conversation, transcripts, and diagnostics.

To experiment safely, edit Assistant Behaviour, watch the token stats update,
then send a short test prompt. If a change makes the assistant worse, use the
prompt history selector to restore an earlier version.

### Single-shot generation

```powershell
python generate.py "Explain quicksort in Python"
python generate.py --think "What is 25 * 37?"
python generate.py --max-tokens 512 --system "You are a poet." "Write a haiku about GPU computing"
```

## VRAM usage

The E2B-it model normally loads in BF16/FP16 and can use about 10 GB VRAM, leaving
roughly 2 GB headroom on a 12 GB card. On GPUs below 12 GB VRAM, the app
automatically tries 4-bit low-VRAM loading instead.

Low-VRAM mode is slower than the normal 12 GB path, but it gives 8 GB cards a
usable route. Keep `Reply length` closer to 512-1024 tokens on 8 GB cards,
especially with Thinking Mode enabled.

GTX 10-series cards such as the GTX 1070 use Pascal compute capability 6.1.
Recent PyTorch CUDA 12.8 wheels do not support that GPU generation, so the setup
assistant switches those cards to the pinned CUDA 11.8 package set in
`requirements-pascal.txt`. If setup reports that the installed PyTorch wheel does
not support the GPU, run:

```powershell
.\scripts\Setup-Gemma4.ps1
```

That reinstalls the compatible package set inside `.venv`.

You can override model loading with the `GEMMA4_LOAD_MODE` environment variable:

```powershell
$env:GEMMA4_LOAD_MODE = "auto"   # default: 4-bit below 12 GB, normal otherwise
$env:GEMMA4_LOAD_MODE = "4bit"   # force low-VRAM quantized loading
$env:GEMMA4_LOAD_MODE = "bf16"   # force normal BF16/FP16 loading
```

The command-line helpers also accept `--load-mode`:

```powershell
python chat.py --load-mode 4bit
python generate.py --load-mode 4bit "Explain quicksort"
```

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

When replacing the app icon, use a clean rebuild so both the embedded executable
icon and the runtime Tk window icon are refreshed:

```powershell
.\scripts\Build.ps1 -Clean
.\scripts\Install-ToPrograms.ps1 -Replace
```

Windows Explorer and taskbar shortcuts may keep showing the previous icon from
the shell icon cache. If the rebuilt `dist\Gemma4Chat\Gemma4Chat.exe` has the
new icon but an installed shortcut still looks old, unpin and repin the app, or
restart Explorer/sign out and back in.

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

Or install to an exact target folder:

```powershell
.\scripts\Install-ToPrograms.ps1 -InstallPath "D:\Tools\Gemma4Chat"
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
current profile's chat logs:

```text
.test.gemma4\diagnostics_YYYYMMDD_HHMMSS.log
```
