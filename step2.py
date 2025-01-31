"""
LangChainエージェントの実装例

このモジュールは、OpenAIのLLMを使用したLangChainエージェントを実装します。
エージェントは以下の機能を提供します：
- 数式の計算
- 現在時刻の取得
- DuckDuckGoを使用したウェブ検索

エージェントは自然言語のクエリを受け取り、適切なツールを選択して応答を生成します。
"""

from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain_openai import OpenAI

load_dotenv()

# Initialize the LLM
llm = OpenAI(temperature=0)


# Define tools
def calculator(expression):
    """
    文字列として与えられた数式を評価し、計算結果を返します。

    Args:
        expression (str): 計算する数式（例: "2 + 2", "3 × 4"）

    Returns:
        Union[float, str]: 計算結果、もしくはエラーメッセージ

    Examples:
        >>> calculator("2 + 2")
        4
        >>> calculator("3 × 4")
        12
    """
    try:
        # Clean up the expression and convert to Python operators
        expression = expression.replace("×", "*")
        expression = expression.replace("÷", "/")

        # Split the expression into parts
        parts = expression.split()
        if len(parts) != 3:
            return "計算エラー: 式は '2 + 2' のような形式で入力してください"

        # Parse numbers and operator
        left = float(parts[0])
        op = parts[1]
        right = float(parts[2])

        # Perform calculation
        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        elif op == "*":
            return left * right
        elif op == "/":
            if right == 0:
                return "計算エラー: ゼロによる除算はできません"
            return left / right
        else:
            return "計算エラー: サポートされていない演算子です"

    except ValueError:
        return "計算エラー: 無効な数値です"
    except (SyntaxError, TypeError):
        return "計算エラー: 無効な式の形式です"


def get_current_time(_):
    """
    現在の日時を日本語フォーマットで返します。

    Args:
        _ (Any): 未使用の引数（LangChainのツール仕様に合わせるため必要）

    Returns:
        str: "YYYY年MM月DD日 HH:MM:SS" 形式の現在日時

    Examples:
        >>> get_current_time(None)
        '2025年02月01日 07:11:33'
    """
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


# Create tool instances
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="数学の計算が必要な場合に使用します。入力は数式の文字列です。",
    ),
    Tool(
        name="CurrentTime",
        func=get_current_time,
        description="現在の日時を取得する必要がある場合に使用します。",
    ),
    Tool(
        name="Search",
        func=search.run,
        description="インターネットで情報を検索する必要がある場合に使用します。",
    ),
]

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# Test the agent with different queries
queries = [
    "2023年のノーベル平和賞受賞者は誰ですか？",
    "今日の日付を教えてください",
    "1234 × 5678 の計算結果を教えてください",
    "東京の現在の人口は何人ですか？具体的な数字で教えてください",
]

print("エージェントのテスト開始:")
print("-" * 50)

for query in queries:
    print(f"\n質問: {query}")
    try:
        response = agent.invoke(query)
        print(f"回答: {response}")
    except (ValueError, KeyError) as e:
        print(f"入力エラー: {str(e)}")
    except (ConnectionError, TimeoutError) as e:
        print(f"ネットワークエラー: {str(e)}")
    except RuntimeError as e:
        print(f"実行時エラー: {str(e)}")
    print("-" * 50)
