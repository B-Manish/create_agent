from fastapi import FastAPI, UploadFile, File, Response
from model_agent import ImageClarifyAgent
from PIL import Image
import io

app = FastAPI()

# Global reusable agent
agent: ImageClarifyAgent = None

# Initialize the agent at server startup
@app.on_event("startup")
def init_agent():
    global agent
    agent = ImageClarifyAgent()

# POST endpoint to accept an image and return a clarified version
@app.post("/clarify-image")
async def clarify_image(file: UploadFile = File(...)):
    # Step 1: Read the image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Step 2: Use the agent to clarify the image
    clarified_image = agent.clarify(image)

    # Step 3: Convert back to bytes to send as response
    buf = io.BytesIO()
    clarified_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    return Response(content=byte_im, media_type="image/png")
