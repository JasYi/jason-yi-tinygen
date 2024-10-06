# Jason Yi Tinygen

## Running Tinygen

run on https://jason-yi-tinygen.vercel.app/

endpoint is on /modify-repo

## Methodology

To handle codebases that often hit the context limit, I created a RAG system to only grab relevant files.

In this system each file is summarized using 4o mini and then the summaries of the files are embedded to run RAG over.

When a user makes a call, runs RAG to find relevant files that it then uses as context to the diff generating call.

The responses to this call is then fed into a second step that creates a summary of the changes and runs a function call if it finds that the changes do not fit the prompt or are in the wrong format.

The function call essentially re-runs this regeneration proecess to correct the mistake.
