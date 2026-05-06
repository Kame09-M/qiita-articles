---
title: DBCV（Density-Based Clustering Validation）とは？？
tags:
  - Python
  - 初心者
  - 機械学習
  - クラスタリング
private: false
updated_at: '2026-04-29T16:47:21+09:00'
id: 8e6afaf66df6dc9267d0
organization_url_name: null
slide: false
ignorePublish: false
---

# DBCV（Density-Based Clustering Validation）

> **一言でいうと**：「HDBSCANなど密度ベースクラスタリング専用の品質指標。シルエットスコアでは正しく評価できないクラスタを、密度の観点から正しく採点できる」

tags: #clustering #evaluation #hdbscan #unsupervised-learning

---

## なぜDBCVが必要か

### シルエットスコアの問題

クラスタリングの評価指標として最も有名な**シルエットスコア**は、クラスタが「球状」であることを前提にしている。

HDBSCANが検出するクラスタは不規則な形状のため、球状前提の指標で評価すると正しく採点できない。

```
シルエットスコアが「良い」と言うクラスタ：
  ●●●    ○○○   ← コンパクトで球状

HDBSCANが実際に検出するクラスタ：
  ●●●●●●●●●●●   ← 細長くても正しいクラスタ
```

DBCVは**密度**を基準にするため、複雑な形状のクラスタでも正しく評価できる。

---

## DBCVが測っていること

2つの観点を同時に評価している。

### ① 凝集度（Compactness）
同じクラスタ内の点が密度的に均一でまとまっているか。

### ② 分離度（Separation）
異なるクラスタの間に、密度の低い「谷」があるか。谷がはっきりしているほど良い。

```
密度
 ↑
 │  ███         ███
 │  ███    谷   ███
 │  ███   ↓↓↓  ███
 │         ↑
 │     ここが分離度の基準
 └──────────────────→ 空間
    クラスタA  クラスタB
```

「クラスタ内が密で、クラスタ間の海が深い」状態が高スコア。

---

## なぜ「距離」ではなく「密度の谷」で分離度を測るのか

距離ベースで分離度を測ると、細長いクラスタの左端と右端が「遠い」として誤って別クラスタ扱いになる。

DBCVの分離度は**2クラスタを結ぶ経路上の「最も密度が低い点」**で決まる。細長いクラスタの内部はずっと密度が高いので、端同士も「つながっている」と正しく判断できる。

---

## スコアの範囲と解釈

| スコア | 意味 |
|--------|------|
| 1.0 | 完璧なクラスタ構造（密で、明確に分離） |
| 0付近 | クラスタ構造がほぼない |
| -1.0 | 最悪（クラスタが完全に混在） |

実際の研究では **0.3〜0.6 あれば十分良い** とされることが多い。

---

## 実装

```python
from hdbscan.validity import validity_index
import numpy as np

# X: クラスタリングに使った特徴量（numpy配列）
# labels: HDBSCANのクラスタラベル（ノイズ点は -1）
dbcv = validity_index(X.astype(np.float64), labels)
print(f"DBCV: {dbcv:.4f}")
```

`float64` へのキャストが必要な点に注意。ノイズ点（ラベル `-1`）は自動でスコア計算から除外される。

---

## パラメータ選択への活用

DBCVを使うと、HDBSCANのパラメータ選択を定量的に正当化できる。

```python
configs = [
    {"min_cluster_size": 100, "min_samples": 10},
    {"min_cluster_size": 200, "min_samples": 10},
    {"min_cluster_size": 300, "min_samples": 20},
]

for cfg in configs:
    c = hdbscan.HDBSCAN(**cfg, metric="euclidean")
    labels = c.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    dbcv = validity_index(X.astype(np.float64), labels)
    print(f"mcs={cfg['min_cluster_size']}, ms={cfg['min_samples']} "
          f"→ クラスタ={n_clusters}, ノイズ率={n_noise/len(labels)*100:.1f}%, DBCV={dbcv:.4f}")
```

### グリッドサーチとの組み合わせ

`min_cluster_size` と `min_samples` を独立して変えながらDBCVを計算することで、
主観に頼らない客観的なパラメータ選択ができる。

```python
import itertools

results = []
for mcs, ms in itertools.product([100, 150, 200, 300], [5, 10, 20, 50]):
    c = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean")
    labels = c.fit_predict(X)
    dbcv = validity_index(X.astype(np.float64), labels)
    results.append({"mcs": mcs, "ms": ms, "dbcv": dbcv})

# DBCVが最高のパラメータを確認
best = max(results, key=lambda x: x["dbcv"])
print(best)
```

### 注意：DBCVが高い ≠ 研究目的に合っている

DBCVは「クラスタの密度的品質」しか測らない。以下は別途確認が必要：

- クラスタ数が多すぎ・少なすぎないか
- ノイズ率が許容範囲か
- クラスタの中身が意味的に解釈できるか

DBCVが最高でもクラスタ数が極端に少なければ分析に使えないケースもある。

---

## 他の評価指標との比較

| 指標 | 密度ベースとの相性 | 特徴 |
|------|-----------------|------|
| **DBCV** | ◎ | 密度ベース専用。不規則形状に対応 |
| シルエットスコア | △ | 球状前提。最も有名だがHDBSCANには不向き |
| Calinski-Harabasz | △ | クラスタ間分散÷クラスタ内分散。球状前提 |
| CDbw | ○ | 密度ベース対応。DBCVより古い手法 |

**結論**：密度ベースクラスタリング（HDBSCAN）の評価にはDBCVが現状ベストプラクティス。

---

## 注意点

- 計算コストが高め。大規模データでは時間がかかる
- `min_samples` を大きくするとDBCVは上がりやすいが、ノイズ率も上がることがある
- `float64` へのキャストを忘れるとエラーになる

---

## 参考

- Moulavi et al. (2014) "Density-Based Clustering Validation" — DBCV原論文
- hdbscan Python ライブラリ公式ドキュメント
- [[HDBSCAN]] — 密度ベースクラスタリング本体
