from dotenv import load_dotenv
import os

load_dotenv()

LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")

OLLAMA_MODEL = "deepseek-r1:7b"
OLLAMA_BASE_URL = "http://localhost:11434"