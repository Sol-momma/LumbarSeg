# T2 SPACE 精度0.97キャンペーン計画

## 結論

論文が報告したスコアへの到達と、論文手順そのものの再現を別々に判定する。T2 SPACEの背景を除く3クラスで精度を判定し、RTX 3060 Tiは1台なので、tmuxで実験を管理しつつGPU学習は1件ずつ順番に実行する。

## 合格条件

### 論文スコアの参照値

| クラス | Dice下限 |
| --- | ---: |
| IVDs（椎間板） | `0.9688` |
| Vertebrae（椎骨） | `0.9712` |
| Spinal Canal（脊柱管） | `0.9671` |

### 精度目標の達成

次を同時に満たした場合だけ成功とする。

1. 3クラスのDice平均（foreground macro Dice）が`0.9700`以上。
2. 3クラスすべてが上表の論文値以上。

Backgroundと既存CSVの`Mean`行は主要判定に使用しない。

この精度条件を満たした状態を`score_target_met`とする。論文と同じ前処理、画像選別、分割、学習、評価であることまで確認できた状態だけを`paper_protocol_verified`とする。改良手法による`score_target_met`を、論文手法そのものの再現成功とは呼ばない。

## 固定する比較条件

- 対象: T2 SPACE。
- 分割: SPIDER Overview CSVのtraining/validationをシリーズ単位で使用。比較期間中は固定する。
- baseline再実行で確定した検証スライス一覧を`fixed_validation_files.txt`として固定し、全候補を同じ一覧で評価する。filtering/samplingの変更は学習側だけへ適用する。
- 検証画像とマスクの論理内容をSHA-256で固定し、候補の学習開始後・評価開始前に一致を必須確認する。前処理自体を比較する場合は、新しい派生表現でbaselineから別キャンペーンを作り直す。
- 検証用の原画像とマスクは固定し、改良手法で書き換えない。向きの変換が必要な場合は、原本を残した派生データ上で対応する画像とマスクへ同じ決定的変換を適用する。マスク修正は学習側だけへ適用する。
- 入力: 512×640の2D矢状断。
- 初期比較条件: 4クラス必須、不均衡しきい値0.90、最大1,000枚、batch size 2、最大100 epoch、seed 42。
- 固定baseline: foreground macro Dice `0.849745`、Vertebrae `0.918222`、Spinal Canal `0.785869`、IVDs `0.845144`。背景込み旧Meanは`0.883624`。
- 比較時は変更点を1種類に限定し、同じ分割、評価コード、seed、出力形式を使う。baseline再実行との差は記録し、無条件に同一結果とみなさない。
- 主要値はIVDs、Vertebrae、Spinal Canalのクラス別Diceとforeground macro Dice。
- IoU、ASD、NSD、Precision、Recall、F1は副次指標として保存する。

## 実験順

| 順番 | 手法 | 目的 | GPU実行 |
| ---: | --- | --- | --- |
| 0 | 現在baselineの再実行 | 新しい判定方法で基準値を固定 | 1件 |
| 1 | 画像方向の監査 | 向き情報と最小軸推定の不一致を学習前に特定 | なし |
| 2 | ラベル品質の監査 | 孤立領域・境界・小面積脊柱管を修正前に定量化 | なし |
| 3 | filtering/sampling比較 | 0.55/0.90と1,000枚選択の影響を分離 | 原則なし |
| 4 | 画像方向の厳密化 | 監査で確認した向きの誤りだけを修正 | 1件 |
| 5 | 学習マスク修正 | 検証マスクを固定し、学習側だけを改善 | 1件 |
| 6 | データ拡張 | 画像方向・位置・濃度変化への頑健性を上げる | 1件 |
| 7 | クラス別損失調整 | 小さい脊柱管と椎間板の学習を強める | 1件 |
| 8 | モデル改良 | 前処理と学習条件で不足する場合のみ検討 | 1件 |

前段で`score_target_met`になった場合は、後続の改良実験を自動停止する。複数変更を同時に入れず、どの変更が精度へ効いたか追跡できる状態を守る。`paper_protocol_verified`は別の再現監査で判定する。

## tmux構成

```text
lumbarseg-goal
├── gpu-queue       学習→評価→目標判定を1件ずつ実行
├── monitor         nvidia-smi、ログ、空き容量を確認
└── cpu-analysis    完了済みCSVの集計だけを並行実行
```

- GPU学習とTensorFlow評価は常に1件だけ実行する。
- CPU処理を並行する場合も別のoutput rootを使用する。
- 長時間実験中は`RECORD_TO_DOCS=0`とし、Git管理下へ自動書き込みしない。
- 出力はキャンペーン日時と実験番号で分離する。
- SSH切断後にtmuxが残ることを短いテストで確認してから開始する。

## 出力

```text
~/lumbarseg_runs/goal_<日時>/
├── 01_t2_space_4cls090_cap1000/
│   ├── checkpoints/
│   ├── run_config.tsv
│   ├── filtered_files.txt
│   ├── validation_metrics.csv
│   └── target_check.json
├── fixed_validation_files.txt
├── fixed_validation_cohort.tsv
├── 02_<次の手法>/
├── logs/
└── campaign_status.tsv
```

各実験で、Git SHA、実コマンド、preset、MRI種類、seed、しきい値、batch/epoch、データ件数、採用slice一覧、train/validation件数、環境、学習ログ、評価CSVを保存する。主要表では`foreground_macro_dice`と背景込みの`legacy_mean_dice`を明確に分ける。

## 停止条件

- `score_target_met`: 後続の改良実験を開始せず正常終了。論文手順の再現完了とは別扱い。
- 目標未達: 次の登録済み手法へ進む。
- 学習失敗、CSV欠損、数値異常: 後続を開始せず異常終了。
- GPU学習が既に動作中: 新しいキャンペーンを開始しない。
- tmux再接続テスト失敗: 長時間実験を開始しない。

## 今回開始する範囲

1. 目標判定処理を追加する。
2. 単一GPU用の順次実行キューを追加する。
3. Windows/WSLのDrive、GPU、tmux、Git版を確認する。
4. 現在baselineを新しい出力先で開始する。
5. baseline開始時のGit SHAをmanifestへ固定する。次の変更は同じcheckoutへ混ぜず、別worktree/branchで画像方向の監査から準備する。
