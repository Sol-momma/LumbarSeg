# 論文条件のベースライン（固定版）

この文書は、論文（Ahmed et al., arXiv 2409.06018）の条件で回すベースラインを1つに決めて固定するためのものです。今後の改善案は、すべてこのベースラインと比べます。
設定ファイルは [`configs/experiments/baseline_protocol.json`](../../configs/experiments/baseline_protocol.json) です（既存の設定と同じ JSON 形式）。数字はすべて実測か、リポジトリ内の記録から取りました。確かめていないことは「未確認」と書いています。

> 状態: 壮宇さんの確認待ちです。この文書に沿った新しい学習（3系列版）は、まだ実行していません。

## 1. 論文の読み方

| 項目 | 論文の記述 | このベースラインでの扱い |
|---|---|---|
| 損失 | Dice + focal。γ は本文に 4.0 と 0.4 の両方が出てくる | **γ 0.4・α 0.6**（α と一緒に書かれている方の値） |
| batch size | 8 | **8** |
| epoch | 100 | **100**（早期終了あり。val mean IoU を監視、patience 15） |
| 画像サイズ | 512×640 | **512×640** |
| Step 1 | 4クラスすべてを含むスライスだけを残す | `min_classes: 4` |
| Step 2 | class weight ratio が 55% を超える画像を除く | 下の「2. 55%フィルター」を参照 |
| 枚数 | フィルター後は T1・T2・T2 SPACE が各1000枚（Table 1） | 系列ごとに1000枚まで（`max_slices_per_sequence: 1000`） |

論文に書かれていない項目（学習率など）は、実装の既定値（`default`）のままです。

## 2. 55%フィルターの解釈

詳しい集計は [paper_filter_check_20261021.md](paper_filter_check_20261021.md) と [paper_filter_all_series_20261021.md](paper_filter_all_series_20261021.md)（PR #14）にあります。要点は次のとおりです。

- **現行の定義 A**（前景3クラスの中で最大のクラスの割合 ≤ 0.55）では、SPIDER 全447系列を合わせても **101枚**（T1 27・T2 40・T2 SPACE 34）しか残りません。論文の「各1000枚」と合いません。
- **A90・C・F の定義**（どれも、4クラスそろうスライスをほぼすべて残す）に**系列ごとの1000枚上限**を組み合わせると、T1 1626・T2 1699・T2 SPACE 1449 枚から各 **1000枚、計3000枚**になり、論文の Table 1 と一致します。
- 論文にある比率の数字（フィルター前 57%・56%・40%、後 55%・55%・37%、別の段落の 3.51%・1.79%・7.5%）は、**どの定義でも再現できません**でした。
- このため、正しい定義は論文からは決められません（著者への確認が必要）。このベースラインでは、枚数が論文と一致する **A90**（実装の既定値 `imbalance_threshold: 0.90`）を採用します。

## 3. 学習・評価の条件（固定値）

| 項目 | 値 | 設定キー |
|---|---|---|
| focal γ / α | 0.4 / 0.6 | `train.focal_gamma` / `train.focal_weight` |
| batch size | 8 | `train.batch_size` |
| epoch | 100 | `train.epochs` |
| 学習率 | 0.0001（論文に記載なし。実装の既定値） | `train.learning_rate` |
| 増強 | なし | `train.augment` |
| seed | **42・43・44**（PR #13 で実際に使った3つ） | `train.seed`（1回ごとに変える） |
| 早期終了用の症例 | 学習用症例から4症例（`inner_val_seed 0`） | `data.inner_val_cases` / `data.inner_val_seed` |
| 主指標 | **症例ごとの Dice**（前景3クラスの平均を症例ごとに出し、症例で平均） | `eval.dice_average_unit: case`、`eval.headline_metric: foreground_macro_dice` |
| 背景 | 指標に含めない | `eval.include_background: false` |

PR #13 では `paper_effective` に `--set data.inner_val_cases=4` と症例単位の評価を指定して回していました。`baseline_protocol.json` はこの2つを設定に書き込み、指定し忘れが起きないようにしています。それ以外は `paper_effective` と同じです。

## 4. テストに使う11症例（T2 SPACE）

SPIDER の Overview CSV（`data/SPIDER Lumbar Spine Segmentation Overview.csv`）で `subset` が `validation` の T2 SPACE 系列です。この症例は学習にも早期終了にも使いません。

`11_t2_SPACE`、`35_t2_SPACE`、`61_t2_SPACE`、`107_t2_SPACE`、`118_t2_SPACE`、`129_t2_SPACE`、`136_t2_SPACE`、`152_t2_SPACE`、`161_t2_SPACE`、`177_t2_SPACE`、`234_t2_SPACE`

- PR #13 の評価は 270スライス・11症例で、`split.sha256=357a8155…`（全回で一致）でした。
- 早期終了用の4症例は、コードの選び方（学習用30症例を並べ、`default_rng(0)` で4つ選ぶ）から計算すると `18_t2_SPACE`、`41_t2_SPACE`、`57_t2_SPACE`、`239_t2_SPACE` になります。ただし、実行時に書き出されたファイル一覧とは**未確認**です（一覧が手元に残っていないため）。

## 5. 使う系列（T1・T2・T2 SPACE）とテスト分割の案

論文に合わせ、次の学習では3系列を使う案です（**未実行**）。

| 系列 | 4クラスそろうスライス | 上限後 | テスト（Overview CSV の validation） | 学習（training） |
|---|---|---|---|---|
| T1 | 1626 | 1000 | 38症例 | 158症例 |
| T2 | 1699 | 1000 | 38症例 | 172症例 |
| T2 SPACE | 1449 | 1000 | 11症例（上の固定リスト） | 30症例 |

- テストは系列ごとに分け、症例 Dice も系列ごとに出します。T2 SPACE の11症例は、これまでの結果と比べられるように変えません。
- 未確認の点: 3系列を1つのモデルで学習するか系列ごとに学習するか、論文の分け方（学習・テストの比）、上限1000枚をかけた後の T1・T2 のテスト枚数。

## 6. 実測した基準値（PR #13）

| 条件 | 系列 | seed | 各回の症例 Dice | 平均 ± SD |
|---|---|---|---|---|
| `paper_effective`（このベースラインと同じ学習条件） | T2 SPACE のみ | 42 / 43 / 44 | 0.8869 / 0.8868 / 0.8867 | **0.8868 ± 0.0001** |

生の記録は [`experiments/results.jsonl`](../../experiments/results.jsonl) の `sP|…`（コミット `5a3b23d`、`dirty=false`）、詳細は [seeds_paper_baseline_20261021.md](seeds_paper_baseline_20261021.md) です。論文の数字（0.97 前後）とは、データの系列も分割も違うため直接比べていません。

## 7. 確かめ方

```bash
python -m unittest tests.test_baseline_protocol -v
python -m lumbarseg_cli train --config baseline_protocol --dry-run   # 学習はしない
```

テストでは、設定が読み込めること、γ・α・batch・epoch・主指標が上の表と同じこと、学習条件が PR #13 の実行と同じこと、テスト11症例が Overview CSV と一致することを確かめます。

## 8. 未確認のこと

- 55%フィルターの正しい定義（論文の比率の数字はどの定義でも再現できない）。
- 3系列版の学習（未実行。費用がかかるため、壮宇さんの OK 待ち）と、その場合のモデルの分け方・テスト枚数。
- 早期終了用4症例の実際のファイル一覧との照合。
- 論文での学習・テストの分け方（Overview CSV の分割と同じかどうか）。
