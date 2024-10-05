from fastapi import FastAPI
from pydantic import BaseModel
from helpers.github_calls import get_repo_files

app = FastAPI()

class Item(BaseModel):
    repoUrl: str
    prompt: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/modify-repo")
async def root(item: Item):
    repo_files = get_repo_files(item.repoUrl)
    return {"message": f"Repo Files: {repo_files}"}