# LangChain Hands-on Tutorial

このリポジトリは、LangChainを使用した実践的なチュートリアルを提供します。OpenAIのLLMを活用した基本的な使用方法から、高度なエージェントの実装まで段階的に学ぶことができます。

## セットアップ

1. 必要なパッケージのインストール:
```bash
pip install langchain langchain-openai python-dotenv
```

2. 環境変数の設定:
`.env`ファイルをプロジェクトのルートディレクトリに作成し、以下の内容を追加してください：
```
OPENAI_API_KEY=your_api_key_here
```

## チュートリアルの内容

### Step 1: 基本的なLLMの使用 (step1.py)

このステップでは、LangChainを使用してOpenAIのLLMに簡単な質問をする方法を学びます：
- OpenAIのLLMインスタンスの作成
- 環境変数の読み込み
- 基本的なプロンプトの送信と応答の取得

```python
from langchain_openai import OpenAI
llm = OpenAI()
response = llm.invoke("今日の天気はどうですか？")
```

### Step 2: エージェントと複数ツールの実装 (step2.py)

このステップでは、より高度な機能を持つLangChainエージェントを実装します：

実装される機能：
- 数式の計算（例：「1234 × 5678」）
- 現在時刻の取得
- DuckDuckGoを使用したウェブ検索

エージェントは以下のツールを使用して、質問に適切に応答します：
- Calculator: 数学的な計算を実行
- CurrentTime: 現在の日時を取得
- Search: インターネットで情報を検索

```python
# エージェントの使用例
queries = [
    "2023年のノーベル平和賞受賞者は誰ですか？",
    "今日の日付を教えてください",
    "1234 × 5678 の計算結果を教えてください",
    "東京の現在の人口は何人ですか？具体的な数字で教えてください",
]
```

## エラーハンドリング

プログラムには以下のエラーハンドリングが実装されています：
- 入力エラー（ValueError, KeyError）
- ネットワークエラー（ConnectionError, TimeoutError）
- 実行時エラー（RuntimeError）

## 注意事項

- OpenAI APIキーの取り扱いには十分注意してください
- APIの利用には料金が発生する可能性があります
- ウェブ検索機能を使用する際はインターネット接続が必要です
