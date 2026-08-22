param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$OutputRoot = "outputs\t2_space_baseline",
    [string]$Sequences = "T2_SPACE",
    [int]$BatchSize = 2,
    [int]$Epochs = 100,
    [int]$Patience = 15,
    [int]$MaxSlicesPerSequence = 1000,
    [int]$Seed = 42,
    [Int64]$MinimumFreeBytes = 20GB,
    [switch]$ForceReprocess
)

$ErrorActionPreference = "Stop"
$PythonExe = ".\.venv\Scripts\python.exe"

function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

function Assert-FreeDiskSpace {
    param([string]$Path, [Int64]$RequiredBytes)
    $FullPath = [IO.Path]::GetFullPath($Path)
    $Drive = New-Object System.IO.DriveInfo([IO.Path]::GetPathRoot($FullPath))
    if ($Drive.AvailableFreeSpace -lt $RequiredBytes) {
        $RequiredGiB = [Math]::Ceiling($RequiredBytes / 1GB)
        $AvailableGiB = [Math]::Round($Drive.AvailableFreeSpace / 1GB, 2)
        throw "At least $RequiredGiB GiB free space is required; found $AvailableGiB GiB."
    }
}

function Assert-BatchHardware {
    param([int]$RequestedBatchSize)
    if ($RequestedBatchSize -lt 8) { return }
    if (!(Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "blocked_hardware: nvidia-smi is required to verify batch size $RequestedBatchSize."
    }
    $MemoryText = ((& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) | Select-Object -First 1).Trim()
    Assert-NativeSuccess "GPU memory query"
    $MemoryMiB = 0
    if (![int]::TryParse($MemoryText, [ref]$MemoryMiB)) {
        throw "blocked_hardware: could not parse GPU memory: $MemoryText"
    }
    # The fixed 8+8 probe OOMed above 8 GiB on the project RTX 3060 Ti. Do not
    # retry a known-impossible paper batch or silently lower it to another run.
    if ($MemoryMiB -lt 12288) {
        throw "blocked_hardware: batch size $RequestedBatchSize requires at least 12288 MiB; detected $MemoryMiB MiB."
    }
}

if (!(Test-Path $PythonExe)) {
    throw "Virtual environment not found. Run scripts\setup_windows_native_gpu.ps1 first."
}
foreach ($RequiredDirectory in @("images", "masks")) {
    if (!(Test-Path -PathType Container (Join-Path $DataRoot $RequiredDirectory))) {
        throw "Dataset directory is missing: $(Join-Path $DataRoot $RequiredDirectory)"
    }
}
if (!(Test-Path -PathType Leaf (Join-Path $DataRoot "SPIDER Lumbar Spine Segmentation Overview.csv"))) {
    throw "SPIDER overview CSV is missing under DataRoot: $DataRoot"
}
if (Test-Path $OutputRoot) {
    # A fresh root is the simplest proof that evaluation cannot load an older
    # checkpoint after a failed or interrupted training process.
    throw "OutputRoot must not already exist. Choose a new experiment directory: $OutputRoot"
}
$DataPath = [IO.Path]::GetFullPath($DataRoot).TrimEnd('\')
$RunPath = [IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
if ($RunPath.Equals($DataPath, [StringComparison]::OrdinalIgnoreCase) -or
    $RunPath.StartsWith($DataPath + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must not be DataRoot or a directory inside it."
}
Assert-FreeDiskSpace -Path $OutputRoot -RequiredBytes $MinimumFreeBytes
Assert-BatchHardware -RequestedBatchSize $BatchSize

$RunStartedAt = [DateTimeOffset]::Now
$LogDirectory = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$TranscriptPath = Join-Path $LogDirectory "windows_train.log"
Start-Transcript -Path $TranscriptPath -Force | Out-Null

try {
    $TensorFlowInfo = (& $PythonExe -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print('tensorflow=' + tf.__version__); print('gpus=' + ','.join(d.name for d in g)); raise SystemExit(0 if g else 'No GPU detected by TensorFlow.')") | Out-String
    Assert-NativeSuccess "TensorFlow GPU preflight"
    Write-Host $TensorFlowInfo.Trim()

    $GitRevision = ((& git rev-parse HEAD) | Out-String).Trim()
    Assert-NativeSuccess "Git revision lookup"
    $PythonVersion = ((& $PythonExe --version 2>&1) | Out-String).Trim()
    Assert-NativeSuccess "Python version lookup"
    $RequirementsHash = (Get-FileHash "requirements-windows-native-gpu.txt" -Algorithm SHA256).Hash.ToLowerInvariant()
    $TensorFlowSummary = $TensorFlowInfo.Trim() -replace "`r?`n", ';'

    $FreezePath = Join-Path $OutputRoot "installed-packages.txt"
    & $PythonExe -m pip freeze | Set-Content -Path $FreezePath -Encoding UTF8
    Assert-NativeSuccess "pip freeze"
    @(
        "key`tvalue",
        "started_at`t$($RunStartedAt.ToString('o'))",
        "git_revision`t$GitRevision",
        "python_executable`t$([IO.Path]::GetFullPath($PythonExe))",
        "python_version`t$PythonVersion",
        "requirements_sha256`t$RequirementsHash",
        "tensorflow_gpu`t$TensorFlowSummary",
        "data_root`t$DataPath",
        "output_root`t$RunPath",
        "sequences`t$Sequences",
        "batch_size`t$BatchSize",
        "epochs`t$Epochs",
        "patience`t$Patience",
        "max_slices_per_sequence`t$MaxSlicesPerSequence",
        "seed`t$Seed"
    ) | Set-Content -Path (Join-Path $OutputRoot "environment_provenance.tsv") -Encoding UTF8

    $trainArgs = @(
        "train.py",
        "--data_root", $DataRoot,
        "--output_root", $OutputRoot,
        "--sequences", $Sequences,
        "--batch_size", $BatchSize,
        "--epochs", $Epochs,
        "--patience", $Patience,
        "--max_slices_per_sequence", $MaxSlicesPerSequence,
        "--seed", $Seed
    )

    if ($ForceReprocess) {
        $trainArgs += "--force_reprocess"
    }

    & $PythonExe @trainArgs
    Assert-NativeSuccess "Training"

    $BestModel = Join-Path $OutputRoot "checkpoints\best_model.keras"
    if (!(Test-Path -PathType Leaf $BestModel)) {
        throw "Training completed without a new best model: $BestModel"
    }
    # OutputRoot was absent at startup, so this timestamp check is a second
    # defence against evaluating a checkpoint copied in by another process.
    if ((Get-Item $BestModel).LastWriteTimeUtc -lt $RunStartedAt.UtcDateTime) {
        throw "Best model predates this run; refusing evaluation: $BestModel"
    }

    & $PythonExe evaluate.py `
        --data_root $DataRoot `
        --output_root $OutputRoot `
        --sequences $Sequences `
        --model_path $BestModel
    Assert-NativeSuccess "Evaluation"

    $MetricsPath = Join-Path $OutputRoot "validation_metrics.csv"
    if (!(Test-Path -PathType Leaf $MetricsPath) -or (Get-Item $MetricsPath).Length -eq 0) {
        throw "Evaluation did not produce a non-empty metrics CSV: $MetricsPath"
    }
}
finally {
    Stop-Transcript | Out-Null
}
