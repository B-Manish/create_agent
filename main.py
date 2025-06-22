from fastapi import FastAPI
from pydantic import BaseModel
from model import say_hello 

app = FastAPI()

class GreetRequest(BaseModel):
    name: str

@app.post("/greet")
def greet(req: GreetRequest):
    output = say_hello(name=req.name)  # ✅ This works now
    return {"message": output}
