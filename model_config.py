import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# OpenAI客户端配置
custom_base_url = os.getenv('OPENAI_BASE_URL', 'http://localhost:13984')
api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

# 创建OpenAI客户端
client = OpenAI(
    base_url=custom_base_url.rstrip('/') + '/api/llm/v1',
    api_key=api_key
)
print(f"Using custom OpenAI base URL: {client.base_url}")

# 默认模型配置
DEFAULT_MODEL = os.getenv('OPENAI_DEFAULT_MODEL', 'qwen2.5-omni-7b-aliyun')
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1500

# 信息提取专用配置
EXTRACTION_MODEL = os.getenv('OPENAI_EXTRACTION_MODEL', 'qwen2.5-omni-7b-aliyun')
EXTRACTION_TEMPERATURE = 0.1
EXTRACTION_MAX_TOKENS = 800

# 流式传输配置
DEFAULT_STREAM = os.getenv('OPENAI_DEFAULT_STREAM', 'false').lower() == 'true'
EXTRACTION_STREAM = os.getenv('OPENAI_EXTRACTION_STREAM', 'false').lower() == 'true'