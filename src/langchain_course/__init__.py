from itertools import chain
from urllib import response

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import os


def main() -> None:
    load_dotenv()
    print("Hello from langchain-course!") 
    print(f"GOOGLE_MODEL_NAME: {os.getenv('GOOGLE_MODEL_NAME')}")

    information = """
    Shivaji was the founder of Maratha dynasty[1] and a warrior king of Maharashtra. He is popularly known as Chhatrapati Shivaji Maharaj. Shivaji was born in the Shivneri Fort in Maharashtra on 19 February 1630 to mother Jijabai and father Shahaji bhosale. It is said that mother Jijabai to goddess Shivai on Shivneri fort for a brave son and kept her son name as Shivaji.[source?]

    At 15 years, he conquered the Torna Fort. The Chakan fort and the Kondana Fort were taken by bribing the Adil Shahi governor. Afzal Khan tried to attempt treachery with Raje Shivaji, Shivaji killed him with Tiger claws. He became the biggest enemy of Mughal Emperor Aurangzeb. He was arrested in Agra Fort by Aurangzeb, but he escaped by using great wits. In 1674 he was coronated.

    In late March 1680, he fell ill with fever and dysentery(blood from excrement), dying around 3–5 April 1680 at the age of 52, on the eve of Hanuman Jayanti. Rumors followed his death, with Muslims believing he had died because of a curse from Jan Muhammad of Jalna[source?]. Some also believe that his second wife, Soyarabai, poisoned him so that his crown might pass to her 10-year-old son Rajaram[source?].

    After his death, the widowed Soyarabai made plans with various ministers of the administration to crown her son Rajaram rather than her prodigal stepson Sambhaji. On 21 April 1680, ten-year-old Rajaram was installed on the throne. However, Sambhaji took possession of the Raigad Fort after killing the commander. On 18 June, he acquired control of Raigad, and formally ascended the throne on 20 July.
    """

    summary_template = """given the information {information} about the person I want you to create
    1. a summary of the information in 3-4 lines
    2. inresting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm=ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL_NAME"), api_key=os.getenv("GOOGLE_API_KEY"),temperature=0)

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    content = response.content
    if isinstance(content, list):
        content = "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    print(content)

if __name__ == "__main__":
    main()