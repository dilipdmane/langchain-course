from dotenv import load_dotenv
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field
load_dotenv()  # take environment variables from .env.

from langchain.agents import create_agent
from langchain_course.helper import get_google_model
from langchain.tools import tool    
from langchain_tavily import TavilySearch

class Source(BaseModel):
    url:str = (Field(..., description="The URL of the source"))
    

class AgentResponse(BaseModel):
    answer:str = (Field(..., description="The answer t the question"))
    sources: list[Source] = (Field(..., description="The sources used by the agent to answer the question"))




def print_agent_response(result: dict) -> None:
    """Print the agent state as readable message records."""
    print("\nAgent response:")
    for index, message in enumerate(result.get("messages", []), start=1):
        role = getattr(message, "type", "unknown")
        content = getattr(message, "content", "")
        print(f"\n{index}. {role}")
        print(f"   Content: {content}        tool_calls = getattr(message, "tool_calls", [])
        if tool_calls:
            print("   Tool calls:")
            for tool_call in tool_calls:
                print(f"   - {tool_call}")



tools = [TavilySearch()]

agent = create_agent(model=get_google_model(), tools=tools,response_format=AgentResponse)

def main():
    print("Invoking Search agent!")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job posting for an Technical product owner in Michigan Detroit Areas and list their details")})
    print_agent_response(result)


if __name__ == "__main__":
    main()