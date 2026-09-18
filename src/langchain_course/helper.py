from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
import os

def get_google_model():
    """
    Returns the Google PaLM model name based on the environment variable.
    """
    print(f"GOOGLE_MODEL_NAME: {os.getenv('GOOGLE_MODEL_NAME')}")

    return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL_NAME"), api_key=os.getenv("GOOGLE_API_KEY"),temperature=0)