from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field

load_dotenv()  # take environment variables from .env.

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_course.helper import get_google_model


class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent = create_agent(
    model=get_google_model(),
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
