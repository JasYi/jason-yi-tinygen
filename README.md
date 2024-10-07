# Jason Yi TinyGen

## Hitting TinyGen

endpoint is at https://jason-yi-tinygen.vercel.app/modify-repo

if it returns a 504 error that means that the process took too long. This is something done by Vercel as functions can only run for a maximum of 300s. This means that the repo may be too big. If this is a problem please email me at jsy37 [at] georgetown [dot] edu.

I have also done my best to speed up this process by parallelizing every step that I can so if this becomes an issue I can quickly upgrade to Vercel Pro.

## Running TinyGen

Run `pip install -r requirements.txt`

Run `fastapi run main.py`

## Methodology

To handle codebases that often hit the context limit, I created a RAG system to only grab relevant files.

In this system each file is summarized using 4o mini and then the summaries of the files are embedded to run RAG over.

When a user makes a call, runs RAG to find relevant files that it then uses as context to the diff generating call.

The responses to this call is then fed into a second step that creates a summary of the changes and runs a function call if it finds that the changes do not fit the prompt or are in the wrong format.

The function call essentially re-runs this regeneration proecess to correct the mistake.
