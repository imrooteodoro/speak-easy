from fastapi import FastAPI, UploadFile
from transformers import pipeline
import torch
import uvicorn

app = FastAPI()
model = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

@app.get("/")
async def index():
     
     return {
          "response" : "hello"
     }
@app.post("/stt")
async def transcrever(audio: UploadFile):
    audio_bytes = await audio.read()
    result = model(audio_bytes)
    return {"texto": result["text"]}


if __name__ == "__main__":
     uvicorn.run(app, host="0.0.0.0", port=8000)