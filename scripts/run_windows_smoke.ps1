param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [Parameter(Mandatory = $true)]
    [string]$TrainFileList,
    [Parameter(Mandatory = $true)]
    [string]$ValidationFileList,
    [string]$OutputRoot = "outputs\t2_space_baseline",
    [string]$RunOutputRoot = "outputs\batch8_smoke",
    [string]$Sequences = "T2_SPACE",
    [int]$BatchSize = 8,
    [int]$Seed = 42,
    [Int64]$MinimumFreeBytes = 2GB
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
    if (!(Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "blocked_hardware: nvidia-smi is required for the batch-size probe."
    }
    $MemoryText = ((& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) | Select-Object -First 1).Trim()
    Assert-NativeSuccess "GPU memory query"
    $MemoryMiB = 0
    if (![int]::TryParse($MemoryText, [ref]$MemoryMiB)) {
        throw "blocked_hardware: could not parse GPU memory: $MemoryText"
    }
    # Prior 8+8 probes exceeded 8 GiB. Preserve the requested paper condition
    # in the evidence and stop instead of silently substituting batch size 2.
    if ($RequestedBatchSize -ge 8 -and $MemoryMiB -lt 12288) {
        throw "blocked_hardware: batch size $RequestedBatchSize requires at least 12288 MiB; detected $MemoryMiB MiB."
    }
}

if (!(Test-Path $PythonExe)) {
    throw "Virtual environment not found. Run scripts\setup_windows_native_gpu.ps1 first."
}
if (!(Test-Path -PathType Container $DataRoot)) {
    throw "DataRoot does not exist: $DataRoot"
}
foreach ($RequiredDirectory in @("images", "masks")) {
    if (!(Test-Path -PathType Container (Join-Path $OutputRoot $RequiredDirectory))) {
        throw "Processed baseline directory is missing: $(Join-Path $OutputRoot $RequiredDirectory)"
    }
}

$ProcessedPath = [IO.Path]::GetFullPath($OutputRoot)
$ProbePath = [IO.Path]::GetFullPath($RunOutputRoot)
if ($ProcessedPath -eq $ProbePath) {
    throw "RunOutputRoot must differ from OutputRoot so smoke artifacts cannot overwrite a full experiment."
}
if ($BatchSize -ne 8) {
    throw "This probe is fixed to batch size 8. Received: $BatchSize"
}
if ($ProbePath.StartsWith($ProcessedPath.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "RunOutputRoot must not be inside OutputRoot."
}
if (Test-Path $RunOutputRoot) {
    throw "RunOutputRoot must not already exist. Choose a new directory: $RunOutputRoot"
}
Assert-FreeDiskSpace -Path $RunOutputRoot -RequiredBytes $MinimumFreeBytes
Assert-BatchHardware -RequestedBatchSize $BatchSize

foreach ($FileList in @($TrainFileList, $ValidationFileList)) {
    if (!(Test-Path -PathType Leaf $FileList)) {
        throw "Smoke cohort file not found: $FileList"
    }
}

# This entrypoint is a memory/compatibility probe, not a shortened accuracy
# experiment. Restricting each cohort to one batch guarantees one train step
# and one validation step, while train.py performs the authoritative duplicate,
# overlap, and image/mask-pair validation before constructing the model.
$TrainCount = @(Get-Content $TrainFileList | Where-Object { $_.Trim() }).Count
$ValidationCount = @(Get-Content $ValidationFileList | Where-Object { $_.Trim() }).Count
if ($TrainCount -ne $BatchSize) {
    throw "TrainFileList must contain exactly $BatchSize non-empty entries. Found: $TrainCount"
}
if ($ValidationCount -ne $BatchSize) {
    throw "ValidationFileList must contain exactly $BatchSize non-empty entries. Found: $ValidationCount"
}

$RunStartedAt = [DateTimeOffset]::Now
New-Item -ItemType Directory -Path (Join-Path $RunOutputRoot "logs") -Force | Out-Null
$TranscriptPath = Join-Path $RunOutputRoot "logs\windows_smoke.log"
Start-Transcript -Path $TranscriptPath -Force | Out-Null

try {
    $TensorFlowInfo = (& $PythonExe -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print('tensorflow=' + tf.__version__); print('gpus=' + ','.join(d.name for d in g)); raise SystemExit(0 if g else 'No GPU detected by TensorFlow.')") | Out-String
    Assert-NativeSuccess "TensorFlow GPU check"
    Write-Host $TensorFlowInfo.Trim()

    $GitRevision = ((& git rev-parse HEAD) | Out-String).Trim()
    Assert-NativeSuccess "Git revision lookup"
    $PythonVersion = ((& $PythonExe --version 2>&1) | Out-String).Trim()
    Assert-NativeSuccess "Python version lookup"
    $RequirementsHash = (Get-FileHash "requirements-windows-native-gpu.txt" -Algorithm SHA256).Hash.ToLowerInvariant()
    $TensorFlowSummary = $TensorFlowInfo.Trim() -replace "`r?`n", ';'
    & $PythonExe -m pip freeze | Set-Content -Path (Join-Path $RunOutputRoot "installed-packages.txt") -Encoding UTF8
    Assert-NativeSuccess "pip freeze"
    @(
        "key`tvalue",
        "started_at`t$($RunStartedAt.ToString('o'))",
        "purpose`tbatch_size_compatibility_probe",
        "git_revision`t$GitRevision",
        "python_executable`t$([IO.Path]::GetFullPath($PythonExe))",
        "python_version`t$PythonVersion",
        "requirements_sha256`t$RequirementsHash",
        "tensorflow_gpu`t$TensorFlowSummary",
        "data_root`t$([IO.Path]::GetFullPath($DataRoot))",
        "processed_root`t$ProcessedPath",
        "run_output_root`t$ProbePath",
        "sequences`t$Sequences",
        "batch_size`t$BatchSize",
        "seed`t$Seed",
        "train_slices`t$TrainCount",
        "validation_slices`t$ValidationCount"
    ) | Set-Content -Path (Join-Path $RunOutputRoot "environment_provenance.tsv") -Encoding UTF8

    $TrainArgs = @(
        "train.py",
        "--data_root", $DataRoot,
        "--output_root", $OutputRoot,
        "--run_output_root", $RunOutputRoot,
        "--sequences", $Sequences,
        "--epochs", "1",
        "--batch_size", $BatchSize,
        "--seed", $Seed,
        "--reuse_processed_only",
        "--train_file_list", $TrainFileList,
        "--validation_file_list", $ValidationFileList
    )

    & $PythonExe @TrainArgs
    Assert-NativeSuccess "Batch-size smoke probe"

    $BestModel = Join-Path $RunOutputRoot "checkpoints\best_model.keras"
    if (!(Test-Path -PathType Leaf $BestModel) -or (Get-Item $BestModel).Length -eq 0) {
        throw "Smoke probe did not produce a non-empty best model: $BestModel"
    }
}
finally {
    Stop-Transcript | Out-Null
}
