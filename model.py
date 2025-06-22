# from agno.agent import Agent
# from agno.models.openai import OpenAIChat
# from agno.tools import tool
# from pprint import pprint

# api_key="tgp_v1_C4s84fmq3Ue2Tap8VU9rVQtX_XwDKO1q2N51G6qiAz0"

# model = OpenAIChat(
#     id="meta-llama/Llama-3-8b-chat-hf",  # ✅ This one works!
#     api_key=api_key,
#     base_url="https://api.together.xyz/v1"  # ✅ Must be set!
# )

# agent = Agent(
#     model=model,
#     description="An agent that greets you using Together AI"
# )
# agent.verbose = True

# @tool(agent)
# def say_hello(name: str) -> str:
#     return f"Hello {name}!"


# if __name__ == "__main__":
#     print(agent.__class__.__module__)
#     response = agent.run("say_hello", name="Manish")
#     # print(response.messages[0].content) 
#     pprint(vars(response))






# api_key = "tgp_v1_C4s84fmq3Ue2Tap8VU9rVQtX_XwDKO1q2N51G6qiAz0"
    
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

api_key = "tgp_v1_C4s84fmq3Ue2Tap8VU9rVQtX_XwDKO1q2N51G6qiAz0"

model = OpenAIChat(
    id="meta-llama/Llama-3-8b-chat-hf",
    api_key=api_key,
    base_url="https://api.together.xyz/v1"
)

agent = Agent(
    model=model,
    description="An agent that greets you using Together AI"
)
agent.verbose = True

@tool(agent)
def say_hello_tool(name: str) -> str:
    return f"Hello {name}!"

# def say_hello(name: str) -> str:
#     res = agent.run("say_hello_tool", name=name)
#     print(res.messages)
#     return res.messages[0].content

def say_hello(name: str) -> str:
    # Use the agent to run the tool
    res = agent.run(f"say_hello_tool", name=name)
    print(f"Agent response: {res}")
    return res.messages[0].content


