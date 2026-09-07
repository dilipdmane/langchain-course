from dotenv import load_dotenv
import os

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")
    print(f"GOOGLE_MODEL_NAME: {os.getenv('GOOGLE_MODEL_NAME')}")


if __name__ == "__main__":
    main()
