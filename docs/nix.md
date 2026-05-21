# Nix Development Environment

このリポジトリでは、Nix を「開発環境の土台」として使います。
Python ライブラリや Node パッケージの実体はリポジトリには保存せず、
`requirements-baseline.txt` と `package-lock.json` から再現します。

## 方針

- Nix: Python、Node.js、ビルド補助ツールなどのバージョンをそろえる
- Python: `.venv` に `requirements-baseline.txt` からインストールする
- JavaScript: `npm install` で `package-lock.json` からインストールする
- Git 管理しないもの: `.venv/`、`node_modules/`、`dist/`、`.astro/`

TensorFlow や SimpleITK は OS、CPU/GPU、Apple Silicon、CUDA の影響を受けやすいため、
Nix に完全固定するよりも、まずは `.venv` と requirements で管理するほうが現実的です。
フル学習は Colab、大学サーバー、研究室 GPU、クラウド GPU 側で同じ requirements を使います。

## 初回セットアップ

Nix shell に入ります。

```bash
nix develop
```

初回実行時に `flake.lock` が生成されます。生成後は、環境を固定するために
`flake.lock` も Git に追加してください。

direnv を使う場合は、リポジトリに入ったとき自動で Nix shell が有効になります。

```bash
direnv allow
```

## Web サイトの実行

```bash
nix develop
npm install
npm run dev
```

ローカルでは以下を開きます。

```text
http://127.0.0.1:4321/LumbarSeg/
```

ビルド確認:

```bash
npm run build
```

## Python ベースラインの実行

Nix shell の中で `.venv` を作ります。

```bash
nix develop
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-baseline.txt
```

短い smoke test から始めます。

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_test \
  --sequences T2_SPACE \
  --epochs 1 \
  --batch_size 2
```

## よく使うコマンド

```bash
nix develop
npm run build
source .venv/bin/activate
python preprocess.py --help
python train.py --help
python evaluate.py --help
```

## なぜライブラリを全部 Nix に入れないか

医学画像セグメンテーションの実験では、TensorFlow、GPU ドライバ、CUDA、Apple Silicon、
Colab 環境の差が大きく出ます。全部を Nix で固定すると再現性は上がりますが、GPU 実行環境で
動かすまでのコストも上がります。

この研究では、まず以下の分担が扱いやすいです。

- Mac: Nix shell で編集、Web、軽い Python 実行
- Colab / GPU: `requirements-baseline.txt` で学習環境を作る
- リポジトリ: 設定ファイル、コード、ドキュメントだけを保存する

必要になった段階で、GPU サーバー用の `flake.nix` や Docker/Nix 統合に拡張できます。
