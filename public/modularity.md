---
title: モジュラリティ（Modularity）
tags:
  - Python
  - 初心者
  - 機械学習
  - クラスタリング
private: false
updated_at: '2026-05-06T15:05:45+09:00'
id: 90b83156e2cef0e08cef
organization_url_name: null
slide: false
ignorePublish: false
---

# モジュラリティ（Modularity）

> **一言でいうと**：「クラスタ内のエッジ密度が、ランダムグラフと比べてどれだけ高いか」を表す指標

---

## 定義

```
Q = Σ [ (クラスタ内のエッジ密度) - (ランダムグラフでの期待エッジ密度) ]
```

- 範囲：-1 〜 1
- Q > 0.3 程度：意味のあるコミュニティ構造あり、とされる
- Q = 0：ランダムグラフと変わらない
- Q < 0：ランダムより悪い構造

---

## 直感的なイメージ

```
良いクラスタリング（Q高い）：
  ●─●─●    ●─●
  └──┘    └─┘
  内部が密   内部が密
     ↕ クラスタ間のエッジが少ない

悪いクラスタリング（Q低い）：
  ●─●─●─●─●  ← クラスタをまたぐエッジが多い
```

---

## 重要な限界：スパースなグラフへのバイアス

### 現象

ネットワーク構築の閾値を上げる（エッジを絞る）ほど、モジュラリティは自動的に上がる。

以下は類似度閾値を変化させた場合の典型的なパターンの例である。

| 閾値（低→高） | エッジ数 | モジュラリティ | 孤立ノード率 |
|-------------|---------|-------------|------------|
| 低い         | 多い    | 0.15 程度   | 5% 程度    |
| 中程度        | 中程度  | 0.30 程度   | 20% 程度   |
| 高い         | 少ない  | 0.50 程度   | 40% 程度   |
| 非常に高い    | 極少    | 0.70 以上   | 60% 以上   |

閾値を上げてノードの大半が孤立した状態でも、モジュラリティは高い値を示す。

### なぜ起きるか

エッジを絞ると「残ったエッジだけで作られるコミュニティ」は必然的にランダム期待値と大きく乖離する。
孤立ノード（1件クラスタ）はエッジを持たないため、ランダム期待値との差分に常にプラスで寄与する。

### 結論

**「モジュラリティが高い = ネットワーク構築の閾値が良い」とは言えない。**

モジュラリティはクラスタリングの内部品質を測るには有効だが、
**ネットワーク構築のパラメータ選択（閾値・percentile）には使えない。**

---

## 何に使えるか

| 用途 | 適切か |
|------|-------|
| 同じネットワークで複数のクラスタリング結果を比較 | ✅ 適切 |
| シード値（seed）を変えた安定性確認 | ✅ 適切 |
| ネットワーク構築の閾値選択 | ❌ 単調増加バイアスがある |
| 異なるアルゴリズム間（例：HDBSCANとLeiden）の比較 | ❌ 空間が異なる |

---

## 代替的なパラメータ選択方法

モジュラリティが使えない場合の選択肢：

1. **別手法のノイズ率・除外率に揃える**
   - 比較対象の手法と同じ除外率を基準にすることで、一貫した比較が可能になる

2. **最近傍類似度の分布を確認する**
   - 孤立ノードの「最も近いノードとの類似度」が閾値直下に集中している場合は、閾値が厳しすぎるサイン

3. **次元削減空間（UMAPなど）での目視確認**
   - クラスタが空間的に分離しているかを確認する

---

## Pythonでの取得方法

```python
import leidenalg

partition = leidenalg.find_partition(
    g,
    leidenalg.ModularityVertexPartition,
    weights="weight",
    seed=42
)

# モジュラリティスコアを取得
modularity = partition.quality()
print(f"モジュラリティ: {modularity:.4f}")
```

---

## 参考文献

- Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community structure in networks. *Physical Review E*, 69(2), 026113. https://doi.org/10.1103/PhysRevE.69.026113
- Fortunato, S. (2010). Community detection in graphs. *Physics Reports*, 486(3–5), 75–174. https://doi.org/10.1016/j.physrep.2009.11.002
- Fortunato, S., & Barthélemy, M. (2007). Resolution limit in community detection. *Proceedings of the National Academy of Sciences*, 104(1), 36–41. https://doi.org/10.1073/pnas.0605965104
- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9(1), 5234. https://doi.org/10.1038/s41598-019-41695-z
