from fastapi import FastAPI
from pydantic import BaseModel
# from helpers.github_calls import get_repo_files
from helpers.ai_calls import prompt_to_diff

app = FastAPI()

class Item(BaseModel):
    repoUrl: str
    prompt: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/modify-repo")
async def root(item: Item):
    reflection, diff = prompt_to_diff(item.repoUrl, item.prompt)
    return diff

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)