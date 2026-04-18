---
title: 埋め込みベクトル（Embedding Vector）
tags:
  - Python
  - 機械学習
  - NLP
  - 自然言語処理
  - 初心者
private: false
updated_at: ''
id: null
organization_url_name: null
slide: false
ignorePublish: false
---

# 埋め込みベクトル（Embedding Vector）

> **一言でいうと**：「テキストの『意味』を数値の配列に変換する技術。意味が似た文章は、数値も近くなる」

---

## テキストをベクトルに変換する仕組み

コンピュータはそのままでは文章を理解できません。文字列を数値に変換する必要があります。
昔は単純な方法（単語の出現回数を数えるBag-of-Wordsなど）が使われていましたが、現代のEmbeddingは**Transformerアーキテクチャ**を使い、文脈ごと意味を捉えます。

### Transformerとアテンション機構

Transformerの核心は**アテンション(注意機構)** です。

```
文章：「銀行に行った」

単語の注意の向き：
　「銀行」→「行った」（動作の対象として注目）
　「行った」→「銀行」（どこへ行ったかを把握）

→ 単語同士の関係を重みとして計算し、文脈を数値に織り込む
```

アテンション機構のおかげで、同じ単語でも文脈によって異なるベクトルが生成されます。

```
「試合に負けて悔しい」の「負けて」
→ スポーツ文脈のベクトル

「借金を負けてもらった」の「負けて」
→ 値引き文脈のベクトル
```

文章全体を処理した最終的な数値の配列が「埋め込みベクトル」です。

---

## Embeddingモデルとは何か

AIモデルにはさまざまな種類があります。

| モデルの種類 | 入力 | 出力 | 例 |
|-------------|------|------|----|
| 言語モデル（LLM） | テキスト | テキスト（生成） | GPT-4, Claude |
| **Embeddingモデル** | **テキスト** | **数値ベクトル** | **text-embedding-3-large** |
| 画像モデル | 画像 | テキスト / ベクトル | CLIP, DALL-E |

LLMは「会話する」モデル、Embeddingモデルは「意味を数値化する」モデルです。用途が全く異なります。

Embeddingモデルは会話ができない代わりに、**意味の近さ（類似度）を数値として計算できる**のが強みです。

---

## なぜ OpenAI の text-embedding-3-large を使うのか

Embeddingモデルは OpenAI 以外にも多数あります。

| モデル | 次元数 | 特徴 |
|--------|--------|------|
| **text-embedding-3-large** | **3,072次元** | 高精度・多言語対応・API経由で使いやすい |
| text-embedding-3-small | 1,536次元 | 軽量・低コスト |
| text-embedding-ada-002 | 1,536次元 | 旧世代・後方互換 |
| multilingual-e5-large | 1,024次元 | オープンソース・多言語 |
| Japanese SimCSE | 768次元 | 日本語特化・無料 |

今回の選択理由：

- 日本語ツイートが対象 → **多言語対応**が必須
- クラスタリング精度を優先 → **高次元（3,072次元）**の方が細かい意味の差を表現できる
- APIで手軽に使える → 実装コストが低い

---

## 文字の類似度との違い

### 「文字が似ている」≠「意味が似ている」

ここが Embedding を学んだときの最初のつまずきポイントです。

```
文字列の類似度で比べると：
　「武器輸出に反対する」と「武器輸出に賛成する」
　→ ほぼ同じ文字列！　類似度 高

Embeddingで比べると：
　「武器輸出に反対する」と「武器輸出に賛成する」
　→ 意味が逆なので距離は離れる

「武器輸出に反対する」と「軍拡に反対する」
　→ 文字は違うが意味が近い → Embeddingの距離は近い
```

テキストマイニングで「意味が似た投稿をまとめたい」場合、文字の一致率ではなく **Embeddingの距離** を使う必要があります。

### 意味を捉えるとはどういうことか

Embeddingは、似た文脈で使われる単語・文章が空間上で近くなるように学習されています。

```
高次元空間のイメージ：

　　・「武器輸出反対」
　　・「軍拡反対」          ← 近い（反戦文脈）
　　・「平和を守れ」

　　　　　　… 遠い …

　　・「防衛費増額は必要だ」
　　・「安全保障を強化せよ」  ← 近い（防衛推進文脈）
```

---

## 活用できる場面

### 1. 類似文書検索
意味が近い文書を探す。キーワードが一致しなくても意味が近ければヒットする。

### 2. クラスタリング
似た内容の文書をグループ化する。UMAPで次元削減してHDBSCANでクラスタリングするのが一般的。

### 3. 異常検知
意味的に浮いている文書を見つける。他の文書と距離が遠い点がノイズや外れ値になる。

### 4. 推薦システム
ユーザーが読んだ記事に意味が近いコンテンツを推薦する。

## 基本的なコード例

```python
from openai import OpenAI

client = OpenAI()

# テキストをベクトルに変換
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="武器輸出に反対する"
)
vector = response.data[0].embedding  # 3072次元のリスト
# 複数テキストを一括変換
texts = ["テキスト1", "テキスト2", "テキスト3"]
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=texts
)
vectors = [r.embedding for r in response.data]
```

## 注意点
- API利用は有料。大量のテキストは事前にベクトル化して`.npy`などにキャッシュしておくと良い
- ベクトルは人間が直接解釈できないのでUMAPなどの次元削減と組み合わせると視覚化できる

## 参考文献

- [OpenAI Embeddings ドキュメント](https://platform.openai.com/docs/guides/embeddings)
- [text-embedding-3-large モデル詳細](https://openai.com/blog/new-embedding-models-and-api-updates)
- [Attention Is All You Need（Transformerの原論文）](https://arxiv.org/abs/1706.03762)
- [Wikipedia - 単語埋め込み](https://ja.wikipedia.org/wiki/%E5%8D%98%E8%AA%9E%E5%9F%8B%E3%82%81%E8%BE%BC%E3%81%BF)
