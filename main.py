from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    repoUrl: str
    prompt: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/modify-repo")
async def root(item: Item):
    return {"message": f"Repo: {item.repoUrl} with prompt: {item.prompt}"}