param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$OutputRoot = "outputs\t2_space_baseline",
    [string]$Sequences = "T2_SPACE",
    [int]$BatchSize = 8,
    [int]$Epochs = 100,
    [int]$Patience = 15,
    [int]$MaxSlicesPerSequence = 1000,
    [switch]$ForceReprocess
)

$ErrorActionPreference = "Stop"
$PythonExe = ".\.venv\Scripts\python.exe"

if (!(Test-Path $PythonExe)) {
    throw "Virtual environment not found. Run scripts\setup_windows_native_gpu.ps1 first."
}

$trainArgs = @(
    "train.py",
    "--data_root", $DataRoot,
    "--output_root", $OutputRoot,
    "--sequences", $Sequences,
    "--batch_size", $BatchSize,
    "--epochs", $Epochs,
    "--patience", $Patience,
    "--max_slices_per_sequence", $MaxSlicesPerSequence
)

if ($ForceReprocess) {
    $trainArgs += "--force_reprocess"
}

& $PythonExe -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPUs:', gpus); raise SystemExit(0 if gpus else 'No GPU detected by TensorFlow.')"
& $PythonExe @trainArgs

$BestModel = Join-Path $OutputRoot "checkpoints\best_model.keras"
if (Test-Path $BestModel) {
    & $PythonExe evaluate.py `
        --data_root $DataRoot `
        --output_root $OutputRoot `
        --sequences $Sequences `
        --model_path $BestModel
}
else {
    Write-Warning "Best model was not found at $BestModel. Skipping evaluation."
}
