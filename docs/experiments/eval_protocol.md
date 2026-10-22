# 評価プロトコル（固定版、2026-10-23）

今後の比較はすべてこのやり方で行います。数字は実測か、リポジトリ内の記録から取りました。確かめていないものは「未確認」と書いています。

## 1. 分割：T2 SPACE・患者ごとの5分割

| 項目 | 値 | 設定キー |
|---|---|---|
| 対象 | SPIDER の T2 SPACE（4クラスがそろうスライス、系列ごとに最大1000枚。[baseline_protocol.md](baseline_protocol.md) と同じ前処理） | `data.sequences=T2_SPACE` |
| 分け方 | Overview CSV の training / validation を合わせ、**患者ごと**に5つへ分ける。同じ患者のスライスが学習とテストに分かれることはない | `data.split=patient_kfold` |
| fold の決め方 | 患者IDを並べ、`numpy.random.default_rng(kfold_seed)` で並べ替え、順に 0,1,2,3,4 と配る（fold の大きさの差は1人以内） | `data.n_folds=5`、`data.kfold_seed=0` |
| 何番目の fold か | 0〜4 | `data.fold` |
| 早期終了用 | 各 fold の学習患者から4人（`data.inner_val_seed=0`）。テスト患者はモデル選択に使わない | `data.inner_val_cases=4` |

- 患者ID自体がデータ由来なので、**分割ファイルはコミットしません**。同じ分割はいつでも `python scripts/make_kfold_split.py --config paper_effective --set data.split=patient_kfold --out /tmp/kfold.json` で作り直せます（リポジトリ内への書き出しは拒否します）。学習時もこの関数（`lumbarseg_cli.cli.kfold_split`）で毎回同じ分割になります。テスト: `tests/test_kfold_protocol.py`。

## 2. 指標

- **主指標：症例ごとの Dice**（`eval.dice_average_unit=case`）。症例の中で全スライスの画素を足して Dice を出し、症例で平均します。
- **クラス別**：椎骨・脊柱管・**椎間板（IVD）**をそれぞれ出し、前景3クラス平均を主な数字にします。
- 表面の指標：ハーネスが既に出している平均表面距離（ASD）と NSD（許容 1 mm）も記録します。
- SPIDER 公式の評価指標：SPIDER チャレンジは Dice のほかに表面距離系の指標を使っていると読めますが、**公式の評価コードとの一致は未確認**です。ここでは上の ASD・NSD で代わりにします。

## 3. 統計

- 単位は fold（5個）。条件ごとに「fold 平均 ± 標準偏差（標本）」を出します。
- 2条件の比較は、同じ fold どうしの差に対する **Wilcoxon の符号順位検定**（両側）。5組では最小の p が 0.0625 なので、**5分割だけでは p<0.05 になりません**。このため、差の大きさと「5 fold すべてで同じ向きか」も合わせて書きます。seed を複数回すときは (fold, seed) を組にして検定します。

## 4. 比べる条件

| 条件 | 設定 | 中身 |
|---|---|---|
| 基準 | `paper_effective` | 論文の設定（batch 8、focal γ0.4・α0.6、100 epoch・早期終了）、T2 SPACE のみ |
| nnU-Net 2D | `nnunet2d`（`NNUNET_CONFIG=paper_effective`） | nnU-Net v2 2D、`nnUNetTrainer_100epochs`、同じ分割 |
| モデルの答えの平均 | `--model 基準.keras,椎間板改善.keras` | 同じ fold の2モデルの softmax を平均（追加学習なし） |
| 椎間板改善 | `ivd_boundary` | 基準に加え、椎間板の境界から2画素以内の focal 損失を5倍にする（`train.boundary_weight=4`） |

## 5. 費用と時間の見積もり（RunPod Secure RTX 4090 $0.89/h）

1条件1foldの時間は、PR #13（[seeds_paper_baseline_20261021.md](seeds_paper_baseline_20261021.md)）の実測から出しました。
- 基準 `paper_effective`：学習 663 / 448 / 519 s（平均 543 s）。podの壁時計は3回で48分 = **1回 約16分**（起動・pip・評価込み）。
- nnU-Net 2D：学習 2678 / 2645 / 2649 s。podの壁時計は3回で155分 = **1回 約52分**。
- 5分割では学習患者が約29人（今は26人）になるので、時間を1.12倍しました（**未確認**）。
- 椎間板改善の時間は未実測です。損失が少し変わるだけなので基準と同じとしました（**未確認**）。
- 平均は学習がなく評価だけなので 1 fold 約5分としました（**未確認**）。

| 条件 | fold数 | 1foldの時間 | 合計時間（台時） | 費用 |
|---|---|---|---|---|
| 基準 | 5 | 0.30 h | 1.49 h | $1.33 |
| nnU-Net 2D | 5 | 0.97 h | 4.87 h | $4.33 |
| モデルの答えの平均 | 5 | 0.08 h（未確認） | 0.42 h | $0.37 |
| 椎間板改善 | 5 | 0.30 h（未確認） | 1.49 h | $1.33 |
| **合計** | 20 | — | **8.27 h** | **$7.36** |

- 並べ方：fold ごとに「基準 → 椎間板改善 → 平均」を1台、nnU-Net を1台で、**10台並列なら約1時間**（一番長い nnU-Net の1fold）。5台並列なら約2時間、1台なら約8.3時間。どの台数でも合計は約 $7.4（＋ボリューム約 $0.01）。
- 予算：開始前の累計 $26.28（2026-10-23 05:53、budget guard の記録）。前回の上限は累計 $28.82（PR #17 のとき）。今回の上限は累計 $40（budget guard を `LIMIT_PCT=80` で再設定）。見込みの終了時累計は約 $33.6 です。
- 各 fold を seed 3回にすると約 $22 になり、上限を超えるので、今回は seed 42 の1回だけです。
