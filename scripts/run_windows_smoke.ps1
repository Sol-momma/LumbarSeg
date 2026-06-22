param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$OutputRoot = "outputs\t2_space_baseline",
    [string]$Sequences = "T2_SPACE",
    [int]$BatchSize = 2
)

$ErrorActionPreference = "Stop"
$PythonExe = ".\.venv\Scripts\python.exe"

if (!(Test-Path $PythonExe)) {
    throw "Virtual environment not found. Run scripts\setup_windows_native_gpu.ps1 first."
}

& $PythonExe -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPUs:', gpus); raise SystemExit(0 if gpus else 'No GPU detected by TensorFlow.')"

& $PythonExe train.py `
    --data_root $DataRoot `
    --output_root $OutputRoot `
    --sequences $Sequences `
    --epochs 1 `
    --batch_size $BatchSize
