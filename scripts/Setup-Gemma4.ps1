param(
    [switch]$CheckOnly,
    [switch]$PreDownloadModel,
    [switch]$Launch
)

$ErrorActionPreference = "Stop"

$ModelId = "google/gemma-4-E2B-it"
$LicenceUrl = "https://huggingface.co/google/gemma-4-E2B-it"
$NvidiaDriverUrl = "https://www.nvidia.com/Download/index.aspx"
$MinimumVramGb = 12
$RecommendedDiskGb = 40

$script:Checks = @()
$script:HardBlocker = $false
$script:SetupNeeded = $false
$script:NextStep = $null

function Add-Check {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Detail,
        [Parameter(Mandatory = $true)]
        [string]$Status,
        [switch]$HardBlocker,
        [switch]$SetupNeeded,
        [string]$NextStep
    )

    $script:Checks += [PSCustomObject]@{
        Name = $Name
        Detail = $Detail
        Status = $Status
    }

    if ($HardBlocker) {
        $script:HardBlocker = $true
    }
    if ($SetupNeeded) {
        $script:SetupNeeded = $true
    }
    if ($NextStep -and -not $script:NextStep) {
        $script:NextStep = $NextStep
    }
}

function Write-Checks {
    Write-Host ""
    Write-Host "Gemma 4 readiness check"
    Write-Host ""

    foreach ($check in $script:Checks) {
        $left = "{0}: {1}" -f $check.Name, $check.Detail
        if ($left.Length -gt 58) {
            $left = $left.Substring(0, 55) + "..."
        }
        Write-Host ("{0,-62} {1}" -f $left, $check.Status)
    }
}

function Get-CommandPath {
    param([Parameter(Mandatory = $true)][string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    return $null
}

function ConvertTo-ProcessArgument {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Argument)

    if ($Argument -eq "") {
        return '""'
    }
    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }

    $result = '"'
    $backslashes = 0
    foreach ($char in $Argument.ToCharArray()) {
        if ($char -eq '\') {
            $backslashes += 1
            continue
        }
        if ($char -eq '"') {
            $result += ('\' * (($backslashes * 2) + 1))
            $result += '"'
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            $result += ('\' * $backslashes)
            $backslashes = 0
        }
        $result += $char
    }
    if ($backslashes -gt 0) {
        $result += ('\' * ($backslashes * 2))
    }
    $result += '"'
    return $result
}

function Join-ProcessArguments {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    return (($Arguments | ForEach-Object { ConvertTo-ProcessArgument -Argument $_ }) -join " ")
}

function Invoke-Capture {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $argumentListProperty = [System.Diagnostics.ProcessStartInfo].GetProperty("ArgumentList")
    if ($argumentListProperty) {
        foreach ($argument in $Arguments) {
            [void]$startInfo.ArgumentList.Add($argument)
        }
    } else {
        $startInfo.Arguments = Join-ProcessArguments -Arguments $Arguments
    }
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    try {
        [void]$process.Start()
    } catch {
        return [PSCustomObject]@{
            ExitCode = -1
            Output = @($_.Exception.Message)
        }
    }
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    $output = @()
    if ($stdout) {
        $output += ($stdout -split "`r?`n")
    }
    if ($stderr) {
        $output += ($stderr -split "`r?`n")
    }

    return [PSCustomObject]@{
        ExitCode = $process.ExitCode
        Output = $output
    }
}

function Test-Python312 {
    param([Parameter(Mandatory = $true)][string]$PyLauncherPath)

    $result = Invoke-Capture -FilePath $PyLauncherPath -Arguments @("-3.12", "-c", "import sys; print(sys.executable); print(sys.version.split()[0])")
    if ($result.ExitCode -ne 0) {
        return $null
    }

    $lines = @($result.Output | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
    if ($lines.Count -lt 2) {
        return $null
    }

    return [PSCustomObject]@{
        Exe = $lines[0]
        Version = $lines[1]
    }
}

function Test-VenvPython {
    param([Parameter(Mandatory = $true)][string]$PythonPath)

    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        return $null
    }

    $result = Invoke-Capture -FilePath $PythonPath -Arguments @("-c", "import sys; print(sys.version.split()[0])")
    if ($result.ExitCode -ne 0) {
        return $null
    }

    $version = @($result.Output | ForEach-Object { "$_".Trim() } | Where-Object { $_ })[0]
    return [PSCustomObject]@{
        Exe = $PythonPath
        Version = $version
    }
}

function Install-Requirements {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,
        [Parameter(Mandatory = $true)]
        [string]$RequirementsPath
    )

    Write-Host ""
    Write-Host "Installing Python dependencies. This can take a while..."
    & $PythonPath -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip."
    }

    & $PythonPath -m pip install -r $RequirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install requirements.txt."
    }
}

function Test-TorchCuda {
    param([Parameter(Mandatory = $true)][string]$PythonPath)

    $code = @"
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
"@
    $result = Invoke-Capture -FilePath $PythonPath -Arguments @("-c", $code)
    if ($result.ExitCode -ne 0) {
        return [PSCustomObject]@{
            Ok = $false
            Detail = "PyTorch import failed"
            Output = ($result.Output -join "`n")
        }
    }

    $lines = @($result.Output | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
    $cudaAvailable = $lines.Count -ge 2 -and $lines[1] -eq "True"
    $detail = if ($cudaAvailable -and $lines.Count -ge 3) { $lines[2] } else { "CUDA unavailable" }
    return [PSCustomObject]@{
        Ok = $cudaAvailable
        Detail = $detail
        Output = ($result.Output -join "`n")
    }
}

function Test-HuggingFaceLogin {
    param([Parameter(Mandatory = $true)][string]$HuggingFaceCliPath)

    $result = Invoke-Capture -FilePath $HuggingFaceCliPath -Arguments @("whoami")
    if ($result.ExitCode -eq 0) {
        $name = @($result.Output | ForEach-Object { "$_".Trim() } | Where-Object { $_ })[0]
        if ($name) {
            return [PSCustomObject]@{ Ok = $true; Detail = $name }
        }
        return [PSCustomObject]@{ Ok = $true; Detail = "logged in" }
    }

    return [PSCustomObject]@{ Ok = $false; Detail = "not logged in" }
}

function Test-IsWindows {
    if (Get-Variable -Name IsWindows -Scope Global -ErrorAction SilentlyContinue) {
        return $IsWindows
    }

    return [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
}

function Invoke-ModelDownload {
    param([Parameter(Mandatory = $true)][string]$PythonPath)

    $code = @"
from huggingface_hub import snapshot_download
snapshot_download("$ModelId")
print("Downloaded or found cached model: $ModelId")
"@

    Write-Host ""
    Write-Host "Checking/downloading Gemma 4 model files..."
    & $PythonPath -c $code
    return $LASTEXITCODE
}

try {
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $venvHfCli = Join-Path $repoRoot ".venv\Scripts\huggingface-cli.exe"
    $requirementsPath = Join-Path $repoRoot "requirements.txt"

    if (-not (Test-IsWindows)) {
        Add-Check -Name "Windows" -Detail "not detected" -Status "NOT RECOMMENDED" -HardBlocker -NextStep "Run this helper on Windows, or use the manual setup path for your platform."
    } else {
        Add-Check -Name "Windows" -Detail "detected" -Status "OK"
    }

    $nvidiaSmi = Get-CommandPath -Name "nvidia-smi"
    if (-not $nvidiaSmi) {
        Add-Check -Name "GPU" -Detail "nvidia-smi not found" -Status "NOT RECOMMENDED" -HardBlocker -NextStep "Install or update the NVIDIA driver, then re-run this script: $NvidiaDriverUrl"
    } else {
        $gpuResult = Invoke-Capture -FilePath $nvidiaSmi -Arguments @("--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
        if ($gpuResult.ExitCode -ne 0) {
            Add-Check -Name "Driver" -Detail "nvidia-smi failed" -Status "NOT RECOMMENDED" -HardBlocker -NextStep "Install or update the NVIDIA driver, then re-run this script: $NvidiaDriverUrl"
        } else {
            $bestGpu = $null
            foreach ($line in $gpuResult.Output) {
                $text = "$line".Trim()
                if (-not $text) { continue }
                $parts = $text -split ","
                if ($parts.Count -lt 2) { continue }
                $memoryMbText = $parts[$parts.Count - 1].Trim()
                $name = (($parts[0..($parts.Count - 2)] -join ",").Trim())
                $memoryMb = 0
                if ([int]::TryParse($memoryMbText, [ref]$memoryMb)) {
                    if (-not $bestGpu -or $memoryMb -gt $bestGpu.MemoryMb) {
                        $bestGpu = [PSCustomObject]@{ Name = $name; MemoryMb = $memoryMb }
                    }
                }
            }

            if (-not $bestGpu) {
                Add-Check -Name "GPU" -Detail "could not read NVIDIA GPU details" -Status "NOT RECOMMENDED" -HardBlocker -NextStep "Check that the NVIDIA driver is installed and that nvidia-smi works in PowerShell."
            } else {
                $vramGb = [math]::Round($bestGpu.MemoryMb / 1024, 1)
                if ($vramGb -lt $MinimumVramGb) {
                    Add-Check -Name "GPU" -Detail ("{0}, {1} GB VRAM" -f $bestGpu.Name, $vramGb) -Status "NOT RECOMMENDED" -HardBlocker -NextStep "Use a machine with an NVIDIA GPU with about 12 GB VRAM for the intended Gemma 4 experience."
                } else {
                    Add-Check -Name "GPU" -Detail ("{0}, {1} GB VRAM" -f $bestGpu.Name, $vramGb) -Status "OK"
                }
                Add-Check -Name "Driver" -Detail "NVIDIA driver detected" -Status "OK"
            }
        }
    }

    $driveRoot = [System.IO.Path]::GetPathRoot($repoRoot.Path)
    $drive = Get-PSDrive -Name $driveRoot.TrimEnd(":\") -ErrorAction SilentlyContinue
    if ($drive) {
        $freeGb = [math]::Round($drive.Free / 1GB, 1)
        if ($freeGb -lt $RecommendedDiskGb) {
            Add-Check -Name "Disk space" -Detail "$freeGb GB free" -Status "ACTION NEEDED" -SetupNeeded -NextStep "Free up disk space before downloading model files and installing dependencies."
        } else {
            Add-Check -Name "Disk space" -Detail "$freeGb GB free" -Status "OK"
        }
    } else {
        Add-Check -Name "Disk space" -Detail "could not check" -Status "UNKNOWN"
    }

    $venvInfo = Test-VenvPython -PythonPath $venvPython
    $venvReady = $venvInfo -and $venvInfo.Version.StartsWith("3.12.")
    $pyLauncher = Get-CommandPath -Name "py"
    $python312 = $null
    if (-not $pyLauncher) {
        if ($venvReady) {
            Add-Check -Name "Python 3.12" -Detail ".venv uses $($venvInfo.Version)" -Status "OK"
        } else {
            Add-Check -Name "Python 3.12" -Detail "py launcher not found" -Status "ACTION NEEDED" -SetupNeeded -NextStep "Install Python 3.12 from python.org, then re-run this script."
        }
    } else {
        $python312 = Test-Python312 -PyLauncherPath $pyLauncher
        if (-not $python312 -or -not $python312.Version.StartsWith("3.12.")) {
            if ($venvReady) {
                Add-Check -Name "Python 3.12" -Detail ".venv uses $($venvInfo.Version)" -Status "OK"
            } else {
                Add-Check -Name "Python 3.12" -Detail "missing" -Status "ACTION NEEDED" -SetupNeeded -NextStep "Install Python 3.12 from python.org, then re-run this script."
            }
        } else {
            Add-Check -Name "Python 3.12" -Detail $python312.Version -Status "OK"
        }
    }

    if ($venvReady) {
        Add-Check -Name "Virtual environment" -Detail ".venv ready" -Status "OK"
    } elseif ($CheckOnly) {
        Add-Check -Name "Virtual environment" -Detail "missing or not Python 3.12" -Status "WILL CREATE" -SetupNeeded -NextStep "Run .\scripts\Setup-Gemma4.ps1 without -CheckOnly to create the virtual environment."
    } elseif ($python312) {
        Write-Host ""
        Write-Host "Creating .venv with Python 3.12..."
        & $pyLauncher -3.12 -m venv (Join-Path $repoRoot ".venv")
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create .venv with Python 3.12."
        }
        $venvInfo = Test-VenvPython -PythonPath $venvPython
        if (-not $venvInfo) {
            throw "Created .venv, but could not run its Python executable."
        }
        Add-Check -Name "Virtual environment" -Detail ".venv created" -Status "OK"
    }

    $torchOk = $false
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $torchCheck = Test-TorchCuda -PythonPath $venvPython
        if ($torchCheck.Ok) {
            $torchOk = $true
            Add-Check -Name "PyTorch CUDA" -Detail $torchCheck.Detail -Status "OK"
        } elseif ($CheckOnly) {
            Add-Check -Name "PyTorch CUDA" -Detail $torchCheck.Detail -Status "PENDING" -SetupNeeded -NextStep "Run .\scripts\Setup-Gemma4.ps1 without -CheckOnly to install dependencies and verify CUDA."
        } elseif (-not $script:HardBlocker) {
            Install-Requirements -PythonPath $venvPython -RequirementsPath $requirementsPath
            $torchCheck = Test-TorchCuda -PythonPath $venvPython
            if ($torchCheck.Ok) {
                $torchOk = $true
                Add-Check -Name "PyTorch CUDA" -Detail $torchCheck.Detail -Status "OK"
            } else {
                Add-Check -Name "PyTorch CUDA" -Detail $torchCheck.Detail -Status "ACTION NEEDED" -SetupNeeded -NextStep "PyTorch installed, but CUDA is unavailable. Update the NVIDIA driver, then re-run this script."
            }
        }
    } else {
        Add-Check -Name "PyTorch CUDA" -Detail "not checked yet" -Status "PENDING" -SetupNeeded
    }

    if (Test-Path -LiteralPath $venvHfCli -PathType Leaf) {
        $hfLogin = Test-HuggingFaceLogin -HuggingFaceCliPath $venvHfCli
        if ($hfLogin.Ok) {
            Add-Check -Name "Hugging Face login" -Detail $hfLogin.Detail -Status "OK"
        } else {
            Add-Check -Name "Hugging Face login" -Detail "not authorised" -Status "ACTION NEEDED" -SetupNeeded -NextStep "Accept the Gemma 4 licence, then run .\.venv\Scripts\huggingface-cli.exe login"
        }
    } else {
        Add-Check -Name "Hugging Face login" -Detail "huggingface-cli unavailable" -Status "PENDING" -SetupNeeded -NextStep "Run .\scripts\Setup-Gemma4.ps1 without -CheckOnly to install dependencies."
    }

    if ($PreDownloadModel -and -not $script:HardBlocker -and $torchOk) {
        $downloadExitCode = Invoke-ModelDownload -PythonPath $venvPython
        if ($downloadExitCode -eq 0) {
            Add-Check -Name "Model files" -Detail "downloaded or cached" -Status "OK"
        } else {
            Add-Check -Name "Model files" -Detail "download failed" -Status "ACTION NEEDED" -SetupNeeded -NextStep "Accept the Gemma 4 licence at $LicenceUrl, login with huggingface-cli, then re-run with -PreDownloadModel."
        }
    } elseif ($PreDownloadModel -and -not $torchOk) {
        Add-Check -Name "Model files" -Detail "not attempted" -Status "PENDING" -SetupNeeded
    }

    Write-Checks

    if ($script:HardBlocker) {
        Write-Host ""
        Write-Host "Result: not recommended"
        Write-Host ""
        Write-Host "Next step:"
        Write-Host "  $script:NextStep"
        exit 2
    }

    if ($script:SetupNeeded) {
        Write-Host ""
        Write-Host "Result: setup needed"
        Write-Host ""
        Write-Host "Gemma 4 requires Hugging Face access. You must personally accept the model licence:"
        Write-Host "  $LicenceUrl"
        if ($script:NextStep) {
            Write-Host ""
            Write-Host "Next step:"
            Write-Host "  $script:NextStep"
        }
        exit 1
    }

    Write-Host ""
    Write-Host "Result: ready"

    if ($Launch) {
        Write-Host ""
        Write-Host "Launching Gemma 4 Chat..."
        & $venvPython (Join-Path $repoRoot "app.py")
    }

    exit 0
} catch {
    Write-Host ""
    Write-Host "Result: unexpected script error"
    Write-Host ""
    Write-Host $_.Exception.Message
    exit 3
}
