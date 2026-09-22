import os

from dotenv import load_dotenv
from google import genai
from langsmith import traceable

gemini_client = genai.Client()

load_dotenv()  # take environment variables from .
model_name = os.getenv("GOOGLE_MODEL_NAME")

MAX_ITERATIONS = 8

tools_for_llm = [
    {
        "name": "get_product_price",
        "description": "Look up the price of a product in the catalog.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "product": {
                    "type": "STRING",
                    "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                },
            },
            "required": ["product"],
        },
    },
    {
        "name": "apply_discount",
        "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "price": {
                    "type": "NUMBER",
                    "description": "The original price"
                },
                "discount_tier": {
                    "type": "STRING",
                    "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    "enum": ["bronze", "silver", "gold"]  # Enum constraint added for tier matching
                },
            },
            "required": ["price", "discount_tier"],
        },
    },
]

@traceable(run_type="tool")
def get_product_price(product: str)->float:
    """Looks up the price of a product"""
    print(f"Looking up price for product: {product}")
    prices = {
        "laptop": 100.0,
        "desktop": 200.0,
        "tablet": 300.0,
    }
    return prices.get(product, 0.0)

@traceable(run_type="tool")
def apply_discout(price:float, discount_tier: str) -> float:
    """Apply Discount tier to the price and calculate final price
    Available discount tiers: bronze, silver, gold
    """
    discount_rates = {
        "bronze": 0.1,
        "silver": 0.2,
        "gold": 0.3,
    }
    discount_rate = discount_rates.get(discount_tier, 0.0)
    return price * (1 - discount_rate)


@traceable(name="google_chat_traced", run_type="llm")
def google_chat_traced(messages):
    return gemini_client.models.generate_content(model=model_name, contents=messages)

@traceable(name="google_agent_loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discout]
    tools_dict = {tool.name: tool for tool in tools}

    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")

    messages=[{"role":"system","content":"You are a helpful shopping assistant. "
         "You have access to a product catalog tool "
         "and a discount tool.\n\n"
         "STRICT RULES — you must follow these exactly:\n"
         "1. NEVER guess or assume any product price. if tools are giving 0 as price. Try to map requested product name with product listed below laptop , desktop, tablet "
         "You MUST call get_product_price first to get the real price.\n"
         "2. Only call apply_discount AFTER you have received "
         "a price from get_product_price. Pass the exact price "
         "returned by get_product_price — do NOT pass a made-up number.\n"
         "3. NEVER calculate discounts yourself using math. "
         "Always use the apply_discount tool.\n"
         "4. If the user does not specify a discount tier, "
         "ask them which tier to use — do NOT assume one."
         "5"},
    {"role": "user", "content": question}]

    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"Iteration {iteration}")
        gemini_client = google_chat_traced(messages)
        ai_messages = gemini_client.candidates[0].content

        tools_calls = ai_messages.tool_calls
        if not tools_calls:
            print(f"\nFinal Answer :{ai_messages.content}")
            return ai_messages.content

        tool_call = tools_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})

        print(f"Tool Name: {tool_name}")
        print(f"Tool Args: {tool_args}")

        tool_call_id = tool_call.get("id")
        tool_to_use= tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool not found: {tool_name}")
        tool_result = tool_to_use.invoke(tool_args)
        print(f"Tool Result: {tool_result}")

        messages.append(ai_messages)
        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call_id))

    print("Error: Max Iterations Reached")
    return None



if __name__ == "__main__":
    print("Hello LangChain Agnt (.blind_tools)!")
    result = run_agent("What is price of ipad after applying gold discount?")
    print(result)
