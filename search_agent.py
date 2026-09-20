from agents import Agent, ModelSettings, Runner
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Agent instructions
instructions = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
"""

agent = Agent(
    name="Research Planner",
    instructions=instructions,
    model="gpt-4.1",                    # Explicitly specify the model to use
    model_settings=ModelSettings(
        temperature=0.0,                # Set to 0 to reduce variability in output
        max_tokens=150,                 # Set to 150 to limit output length
        top_p=1.0,                      # Set to 1.0 to consider all tokens
        frequency_penalty=0.5,          # Set to 0.5 to reduce repetition
        presence_penalty=0.5,           # Set to 0.5 to encourage new topics
    )
)

input = "Learn about AI agents"

result = Runner.run_sync(
    agent, 
    input=input,
)

print(result.final_output)