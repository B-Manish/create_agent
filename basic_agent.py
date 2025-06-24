from dotenv import load_dotenv
from agno.agent import Agent, RunResponse  # noqa
from agno.models.groq import Groq
from agno.app.fastapi.app import FastAPIApp
from agno.tools import tool

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import os
import random

load_dotenv()
api_key= os.getenv("GROQ_API_KEY")

@tool(show_result=True, stop_after_tool_call=True)
def get_weather(city: str) -> str:
    """Get the weather of a city"""
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    random_weather = random.choice(weather_conditions)

    return f"The weather in {city} is {random_weather}."

@tool(show_result=True, stop_after_tool_call=True)
def get_state(city: str) -> str:
    """ Get the state in which the city is located"""
    states={"Hyderabad":"Telangana","Chennai":"Tamil Nadu"}

    return f"{city} is in {states[city]}."    

basic_agent = Agent(
    name="basic_agent",
    model=Groq(id="llama-3.3-70b-versatile",api_key=api_key), 
    tools=[get_weather,get_state],
    markdown=True)


# basic_agent.print_response("How is the weather in chennai?")

# # Get the response in a variable
# # run: RunResponse = basic_agent.run("Share a 2 sentence horror story")
# # print(run.content)


# fastapi_app = FastAPIApp(
#     agents=[basic_agent],
#     name="Basic Agent",
#     app_id="basic_agent",
#     description="A basic agent that can answer questions and help with tasks.",
# )

# app = fastapi_app.get_app()

# if __name__ == "__main__":
#     fastapi_app.serve(app="basic_agent:app", port=8001, reload=True)


# app = FastAPI()

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/ask")
# async def ask_agent(request: QueryRequest):
#     try:
#         run = basic_agent.run(request.query)
#         return {"response": run.content}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))











# # from agno.agent import Agent
# # from agno.models.groq import Groq
# # from agno.tools.duckduckgo import DuckDuckGoTools
# # from agno.tools.newspaper4k import Newspaper4kTools

# # agent = Agent(
# #     model=Groq(id="llama-3.3-70b-versatile",api_key=api_key),
# #     tools=[DuckDuckGoTools(), Newspaper4kTools()],
# #     description="You are a senior NYT researcher writing an article on a topic.",
# #     instructions=[
# #         "For a given topic, search for the top 5 links.",
# #         "Then read each URL and extract the article text, if a URL isn't available, ignore it.",
# #         "Analyse and prepare an NYT worthy article based on the information.",
# #     ],
# #     markdown=True,
# #     show_tool_calls=True,
# #     add_datetime_to_instructions=True,
# # )
# # agent.print_response("Simulation theory", stream=True)





# # from agno.agent import Agent
# # from agno.media import Image
# # from agno.models.groq import Groq

# # agent = Agent(model=Groq(id="llama-3.2-90b-vision-preview",api_key=api_key))

# # agent.print_response(
# #     "Tell me about this image",
# #     images=[
# #         Image(url="https://upload.wikimedia.org/wikipedia/commons/f/f2/LPU-v1-die.jpg"),
# #     ],
# #     stream=True,
# # )


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
from model import predict, load_weights

app = FastAPI()

# Load weights on startup
load_weights("trained_weights.npz")

@app.post("/predict-digit/")
async def predict_digit(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L")  # grayscale
        img = img.resize((28, 28))
        img_array = np.array(img)

        input_vector = img_array.flatten().reshape(1, 784) / 255.0
        digit = predict(input_vector)
        return JSONResponse(content={"digit": digit})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
