param(
    [string]$PythonCommand = "python",
    [string]$VenvPath = ".venv",
    [Int64]$MinimumFreeBytes = 10GB
)

$ErrorActionPreference = "Stop"

function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

function Assert-FreeDiskSpace {
    param([string]$Path, [Int64]$RequiredBytes)
    $FullPath = [IO.Path]::GetFullPath($Path)
    $Root = [IO.Path]::GetPathRoot($FullPath)
    $Drive = New-Object System.IO.DriveInfo($Root)
    if ($Drive.AvailableFreeSpace -lt $RequiredBytes) {
        $RequiredGiB = [Math]::Ceiling($RequiredBytes / 1GB)
        $AvailableGiB = [Math]::Round($Drive.AvailableFreeSpace / 1GB, 2)
        throw "At least $RequiredGiB GiB free space is required on $Root; found $AvailableGiB GiB."
    }
}

if (Test-Path $VenvPath) {
    # Reusing a partially upgraded environment makes an exact requirements file
    # insufficient evidence. Require a new venv so pip cannot retain stale wheels.
    throw "VenvPath already exists. Remove it explicitly or choose a new path: $VenvPath"
}
if (!(Test-Path -PathType Leaf "requirements-windows-native-gpu.txt")) {
    throw "requirements-windows-native-gpu.txt was not found. Run setup from the repository root."
}

Assert-FreeDiskSpace -Path $VenvPath -RequiredBytes $MinimumFreeBytes

Write-Host "Checking the required Python 3.10 interpreter"
& $PythonCommand -c "import sys; print(sys.version); raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 'Native Windows TensorFlow GPU requires Python 3.10.')"
Assert-NativeSuccess "Python version check"

Write-Host "Creating virtual environment at $VenvPath"
& $PythonCommand -m venv $VenvPath
Assert-NativeSuccess "Virtual environment creation"

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
if (!(Test-Path $PythonExe)) {
    throw "Python executable was not created: $PythonExe"
}

$SetupLogPath = Join-Path $VenvPath "windows_setup.log"
Start-Transcript -Path $SetupLogPath -Force | Out-Null
try {
    Write-Host "Upgrading pip"
    & $PythonExe -m pip install --upgrade pip setuptools wheel
    Assert-NativeSuccess "pip bootstrap"

    Write-Host "Installing native Windows GPU requirements"
    & $PythonExe -m pip install -r requirements-windows-native-gpu.txt
    Assert-NativeSuccess "Pinned requirement installation"

    Write-Host "Checking installed dependency consistency"
    & $PythonExe -m pip check
    Assert-NativeSuccess "pip check"

    Write-Host "Checking TensorFlow GPU visibility"
    $TensorFlowInfo = (& $PythonExe -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print('tensorflow=' + tf.__version__); print('gpus=' + ','.join(d.name for d in gpus)); raise SystemExit(0 if gpus else 'No GPU detected by TensorFlow.')") | Out-String
    Assert-NativeSuccess "TensorFlow GPU check"
    Write-Host $TensorFlowInfo.Trim()

    # The frozen package list and requirement hash make later experiment evidence
    # distinguish a rebuilt environment from one that only happens to import.
    $FreezePath = Join-Path $VenvPath "installed-packages.txt"
    & $PythonExe -m pip freeze | Set-Content -Path $FreezePath -Encoding UTF8
    Assert-NativeSuccess "pip freeze"
    $RequirementsHash = (Get-FileHash "requirements-windows-native-gpu.txt" -Algorithm SHA256).Hash.ToLowerInvariant()
    $PythonVersion = ((& $PythonExe --version 2>&1) | Out-String).Trim()
    Assert-NativeSuccess "Python provenance check"
    $TensorFlowSummary = $TensorFlowInfo.Trim() -replace "`r?`n", ';'
    $ProvenancePath = Join-Path $VenvPath "environment_provenance.tsv"
    @(
        "key`tvalue",
        "created_at`t$([DateTimeOffset]::Now.ToString('o'))",
        "python_executable`t$([IO.Path]::GetFullPath($PythonExe))",
        "python_version`t$PythonVersion",
        "requirements_sha256`t$RequirementsHash",
        "os_version`t$([Environment]::OSVersion.VersionString)",
        "tensorflow_gpu`t$TensorFlowSummary"
    ) | Set-Content -Path $ProvenancePath -Encoding UTF8
}
finally {
    Stop-Transcript | Out-Null
}

Write-Host ""
Write-Host "Setup complete. Activate with:"
Write-Host "  .\$VenvPath\Scripts\Activate.ps1"
Write-Host "Environment evidence: $ProvenancePath"
Write-Host "Setup log: $SetupLogPath"
