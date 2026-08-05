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

途中で依存関係を入れ直す場合は、古い `.venv` を削除してから再実行します。

```powershell
Remove-Item -Recurse -Force .\.venv
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_native_gpu.ps1
```

GPU が見えているかだけ確認したい場合は、次の出力に `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')` のような表示が出ることを確認します。

```powershell
.\.venv\Scripts\python.exe -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

1 epoch の smoke test を実行します。学習用・検証用の固定一覧を必ず指定し、既存の本実験とは別の出力先を使います。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows_smoke.ps1 `
  -DataRoot "D:\SPIDER\DataSet" `
  -OutputRoot "outputs\t2_space_baseline" `
  -RunOutputRoot "outputs\batch8_probe_001" `
  -TrainFileList "cohorts\batch8_train.txt" `
  -ValidationFileList "cohorts\batch8_validation.txt" `
  -BatchSize 8
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
python evaluate.py --data_root /mnt/d/SPIDER/DataSet --output_root outputs/t2_space_baseline --model_path outputs/t2_space_baseline/checkpoints/best_model.keras
```

### WSLでbatch size 8だけを安全に確認する

これは精度実験ではなく、GPUメモリ上でbatch size 8が1回の学習と検証を完了できるかだけを確認する試験です。既存の前処理済みデータを読み取りますが、既存のモデルや結果は上書きしません。

まず、固定baselineの一覧から8件ずつを新しい確認用フォルダへコピーします。

```bash
mkdir -p ~/lumbarseg_runs/cohorts
sed -n '1,8p' ~/lumbarseg_runs/goal_t2_space_baseline_20260805/01_t2_space_4cls090_cap1000/train_files.txt \
  > ~/lumbarseg_runs/cohorts/batch8_train.txt
sed -n '1,8p' ~/lumbarseg_runs/goal_t2_space_baseline_20260805/01_t2_space_4cls090_cap1000/validation_files.txt \
  > ~/lumbarseg_runs/cohorts/batch8_validation.txt
```

次に、既存結果と異なる新しい出力先を指定して実行します。

実行前に、TensorFlowを入れた環境を有効化し、`command -v python`で使用するPythonを確認してください。上の手順どおりリポジトリ内へ作った場合は次です。

```bash
source .venv/bin/activate
command -v python
```

```bash
DATA_ROOT="/mnt/g/My Drive/DataSet" \
PROCESSED_ROOT="$HOME/lumbarseg_runs/goal_t2_space_baseline_20260805/01_t2_space_4cls090_cap1000" \
RUN_OUTPUT_ROOT="$HOME/lumbarseg_runs/batch8_probe_001" \
TRAIN_FILE_LIST="$HOME/lumbarseg_runs/cohorts/batch8_train.txt" \
VALIDATION_FILE_LIST="$HOME/lumbarseg_runs/cohorts/batch8_validation.txt" \
BATCH_SIZE=8 \
PYTHON_EXECUTABLE="$(command -v python)" \
bash scripts/run_wsl_batch_probe.sh
```

成功条件は、GPUが検出され、1 epochの学習と検証がエラーなく終了することです。ここで得た精度は8件だけの確認値なので、0.97判定には使いません。

### 論文との差を調べる監査

監査は生データを読み取るだけで、画像やマスクを変更しません。出力先にはCSV、確認用画像、方向補正案だけを新規作成します。

```bash
python scripts/audit_data_alignment.py \
  --data_root "/mnt/g/My Drive/DataSet" \
  --output_dir "$HOME/lumbarseg_runs/audits/data_alignment_001"

python scripts/audit_filter_sensitivity.py \
  --data_root "/mnt/g/My Drive/DataSet" \
  --processed_root "$HOME/lumbarseg_runs/goal_t2_space_baseline_20260805/01_t2_space_4cls090_cap1000" \
  --output_dir "$HOME/lumbarseg_runs/audits/filter_sensitivity_001" \
  --thresholds "0.55,0.90,1.0"
```

方向補正案は初期状態では未確認扱いです。人が確認済みに変更するまで、学習には使用できません。
