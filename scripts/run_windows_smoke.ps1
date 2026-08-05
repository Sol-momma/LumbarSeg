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
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"
$PythonExe = ".\.venv\Scripts\python.exe"

if (!(Test-Path $PythonExe)) {
    throw "Virtual environment not found. Run scripts\setup_windows_native_gpu.ps1 first."
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

& $PythonExe -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPUs:', gpus); raise SystemExit(0 if gpus else 'No GPU detected by TensorFlow.')"
if ($LASTEXITCODE -ne 0) {
    throw "TensorFlow GPU check failed with exit code $LASTEXITCODE."
}

$TrainArgs = @(
    "train.py",
    "--data_root", $DataRoot,
    "--output_root", $OutputRoot,
    "--run_output_root", $RunOutputRoot,
    "--sequences", $Sequences,
    "--epochs", "1",
    "--batch_size", $BatchSize,
    "--reuse_processed_only",
    "--train_file_list", $TrainFileList,
    "--validation_file_list", $ValidationFileList
)

& $PythonExe @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Batch-size smoke probe failed with exit code $LASTEXITCODE."
}
