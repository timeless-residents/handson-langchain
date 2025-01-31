from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAIのLLMインスタンスを作成
llm = OpenAI()

# LLMに問い合わせる
prompt = "今日の天気はどうですか？"
response = llm.invoke(prompt)

print("LLMの返答:")
print(response)
