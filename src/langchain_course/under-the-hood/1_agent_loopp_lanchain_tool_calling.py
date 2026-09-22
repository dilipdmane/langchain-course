from dotenv import load_dotenv
from langchain_course.helper import get_google_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()  # take environment variables from .
get_google_model()
MAX_ITERATIONS = 8

@tool
def get_product_price(product: str)->float:
    """Looks up the price of a product"""
    print(f"Looking up price for product: {product}")
    prices = {
        "laptop": 100.0,
        "desktop": 200.0,
        "tablet": 300.0,
    }
    return prices.get(product, 0.0)
@tool
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

@traceable(name="langchain_agent_loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discout]
    tools_dict = {tool.name: tool for tool in tools}
    llm = get_google_model()
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")

    messages=[SystemMessage(content="You are a helpful shopping assistant. "
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
         "5"
    ), HumanMessage(content=question)]

    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"Iteration {iteration}")
        ai_messages = llm_with_tools.invoke(messages)

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
