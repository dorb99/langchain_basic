import os
import dotenv

dotenv.load_dotenv("../.env")

OLLAMA_MODEL = os.getenv("ROUTER_OLLAMA_MODEL", "llama3.2:3b")
ROUTER_TEMPERATURE = float(os.getenv("ROUTER_ROUTER_TEMPERATURE", "0"))
HANDLER_TEMPERATURE = float(os.getenv("ROUTER_HANDLER_TEMPERATURE", "0.2"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")