from langchain.agents import initialize_agent, AgentType, Tool
from langchain.utilities import WikipediaAPIWrapper
from gemini_llm import GeminiLLM

def create_agent():
    llm = GeminiLLM()

    wiki = WikipediaAPIWrapper()

    tools = [
        Tool(
            name="Wikipedia",
            func=wiki.run,
            description="Useful for answering questions about general topics."
        )
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent

agent = create_agent()

def generate_blog(topic: str) -> str:
    prompt = f"""You are a professional blog writer. 
Generate a detailed and informative blog on the topic: "{topic}". 
Your blog must include the following sections:

1. Title
2. Introduction
3. Main Content (3-4 paragraphs)
4. Summary

Use information from Wikipedia and other sources if needed. Keep the tone engaging and factual.


Answer the following question. 
Only return either:
- A final answer in the format: 'Final Answer: ...'
OR
- An action and input in the format:
    Action: ...
    Action Input: ...
Do not return both.

"""
    result = agent.invoke({"input": prompt})
    return result["output"]
