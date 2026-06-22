# Windows GPU 実行手順

この手順は、Windows PC の NVIDIA GPU で baseline を実行するための最短手順です。

## 推奨構成

TensorFlow 公式ドキュメントでは、native Windows の GPU 対応は `tensorflow<2.11` が最後です。新しい TensorFlow を使う場合は WSL2 が推奨です。参考: [Install TensorFlow with pip](https://www.tensorflow.org/install/pip)

このリポジトリでは、貸与 Windows PC でそのまま試せるように native Windows GPU 用の手順を用意しています。

## 事前準備

Windows 側で次を入れてください。

- NVIDIA Driver
- Miniconda または Python 3.10
- CUDA 11.2
- cuDNN 8.1
- Git

SPIDER データセットは次の構成で置きます。

```text
D:\SPIDER\DataSet\
├── images\
├── masks\
└── SPIDER Lumbar Spine Segmentation Overview.csv
```

## 実行コマンド

PowerShell を開き、リポジトリを取得します。

```powershell
git clone https://github.com/Sol-momma/LumbarSeg.git
cd LumbarSeg
```

仮想環境を作り、Windows native GPU 用の依存関係を入れます。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_native_gpu.ps1
```

GPU が見えているかだけ確認したい場合は、次の出力に `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')` のような表示が出ることを確認します。

```powershell
.\.venv\Scripts\python.exe -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

1 epoch の smoke test を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows_smoke.ps1 -DataRoot "D:\SPIDER\DataSet"
```

本学習と評価を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows_train.ps1 -DataRoot "D:\SPIDER\DataSet"
```

出力先はデフォルトで `outputs\t2_space_baseline` です。学習済みモデルは `outputs\t2_space_baseline\checkpoints\best_model.keras` に保存されます。

## よく使う変更

出力先を変える場合:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows_train.ps1 -DataRoot "D:\SPIDER\DataSet" -OutputRoot "E:\LumbarSegRuns\t2_space"
```

VRAM が不足する場合:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows_train.ps1 -DataRoot "D:\SPIDER\DataSet" -BatchSize 4
```

前処理を作り直す場合:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows_train.ps1 -DataRoot "D:\SPIDER\DataSet" -ForceReprocess
```

全 slice を使う場合:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows_train.ps1 -DataRoot "D:\SPIDER\DataSet" -MaxSlicesPerSequence 0
```

## WSL2 で実行する場合

WSL2 で実行できるなら、native Windows より新しい TensorFlow を使えます。

```bash
git clone https://github.com/Sol-momma/LumbarSeg.git
cd LumbarSeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "tensorflow[and-cuda]" SimpleITK opencv-python numpy pandas scipy scikit-learn tqdm matplotlib
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python train.py --data_root /mnt/d/SPIDER/DataSet --output_root outputs/t2_space_baseline --sequences T2_SPACE --epochs 1 --batch_size 2
python train.py --data_root /mnt/d/SPIDER/DataSet --output_root outputs/t2_space_baseline --sequences T2_SPACE --batch_size 8 --epochs 100
python evaluate.py --data_root /mnt/d/SPIDER/DataSet --output_root outputs/t2_space_baseline --model_path outputs/t2_space_baseline/checkpoints/best_model.keras
```
