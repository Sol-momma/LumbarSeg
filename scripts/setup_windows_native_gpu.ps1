param(
    [string]$PythonCommand = "python",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment at $VenvPath"
& $PythonCommand -m venv $VenvPath

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
if (!(Test-Path $PythonExe)) {
    throw "Python executable was not created: $PythonExe"
}

Write-Host "Upgrading pip"
& $PythonExe -m pip install --upgrade pip setuptools wheel

Write-Host "Installing native Windows GPU requirements"
& $PythonExe -m pip install -r requirements-windows-native-gpu.txt

Write-Host "Checking TensorFlow GPU visibility"
& $PythonExe -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('TensorFlow:', tf.__version__); print('GPUs:', gpus); raise SystemExit(0 if gpus else 'No GPU detected by TensorFlow.')"

Write-Host ""
Write-Host "Setup complete. Activate with:"
Write-Host "  .\$VenvPath\Scripts\Activate.ps1"
