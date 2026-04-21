# TestGemma4

Test Google's **Gemma 4 E2B-it** model locally on a suitable NVIDIA GPU.

## Quick Windows setup

Gemma 4 is a local model, so the computer needs a suitable NVIDIA GPU. This app
targets about **12 GB VRAM** for the intended experience.

Before running the app for the first time:

1. Download or clone this project.
2. Open PowerShell in the project folder.
3. Create or use a [Hugging Face](https://huggingface.co/) account.
4. Open the [Gemma 4 model page](https://huggingface.co/google/gemma-4-E2B-it)
   and accept the model licence.
5. Run the setup assistant:

```powershell
.\scripts\Setup-Gemma4.ps1
```

The setup assistant checks the GPU, NVIDIA driver, disk space, Python
environment, PyTorch CUDA support, and Hugging Face login. It creates `.venv`
and installs the Python dependencies when needed.

If the assistant prints `ACTION NEEDED`, follow the final `Next step` it shows,
then run the same command again.

After Hugging Face access is ready, you can pre-download the model:

```powershell
.\scripts\Setup-Gemma4.ps1 -PreDownloadModel
```

To launch the app after a successful setup:

```powershell
.\scripts\Setup-Gemma4.ps1 -Launch
```

To only check whether the computer looks suitable, without changing setup:

```powershell
.\scripts\Setup-Gemma4.ps1 -CheckOnly
```

The assistant cannot create a Hugging Face account, accept the Gemma 4 licence
for you, install NVIDIA drivers, or install Python globally. It will tell you
when one of those manual steps is required.

## Manual developer setup

```powershell
# Create a venv with Python 3.12 (using py launcher)
py -3.12 -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Log in to Hugging Face (needed once for gated model access)
huggingface-cli login
```

The Hugging Face account must have accepted the Gemma model licence before the
first model download.

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
