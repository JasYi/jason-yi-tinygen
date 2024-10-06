from helpers.github_calls import get_repo_files
import os
import openai
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple, List
import numpy as np

def summarize_code(filename: str, code: str, model: str = "gpt-4o-mini") -> str:
    """
    Generates a summary for a given code file using OpenAI's Chat Completion API.

    :param filename: Name of the file.
    :param code: Content of the file.
    :param model: OpenAI model to use for summarization.
    :param max_retries: Maximum number of retries in case of API failure.
    :param backoff_factor: Factor by which the wait time increases after each retry.
    :return: Summary of the code.
    """
    prompt = (
        f"Analyze the following code in '{filename}' and provide a concise summary of its functionality, "
        "including key components and their interactions.\n\n"
        f"```{filename}\n{code}\n```"
    )

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant proficient in analyzing and summarizing code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500  # Adjust based on desired summary length
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")

def embed_text(text: str, model: str = "text-embedding-3-small") -> list:
    """
    Generates an embedding for the given text using OpenAI's Embedding API.

    :param text: Text to embed.
    :param model: OpenAI embedding model to use.
    :param max_retries: Maximum number of retries in case of API failure.
    :param backoff_factor: Factor by which the wait time increases after each retry.
    :return: Embedding vector as a list of floats.
    """
    try:
        response = openai.embeddings.create(
            model=model,
            input=text
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")

def process_file(filename_code_tuple: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
    filename, code = filename_code_tuple
    try:
        summary = summarize_code(filename, code)
        embedding = embed_text(summary)
        result = {
            "summary": summary,
            "embedding": embedding
        }
        print(f"Successfully processed '{filename}'.\n")
    except Exception as e:
        print(f"Error processing '{filename}': {e}\n")
        result = {
            "summary": None,
            "embedding": None,
            "error": str(e)
        }
    return filename, result

def analyze_codebase(codebase: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Analyzes a codebase by summarizing each file and generating embeddings for the summaries.

    :param codebase: Dictionary with filenames as keys and code content as values.
    :return: Dictionary with filenames as keys and dictionaries containing 'summary' and 'embedding' as values.
    """
    with ThreadPoolExecutor() as executor:
        # Map the process_file function to each item in the codebase
        results = dict(executor.map(process_file, codebase.items()))
    return results

def compute_cosine_similarity(vec1: List, vec2: List) -> float:
    """
    Computes the cosine similarity between two vectors.

    :param vec1: First vector.
    :param vec2: Second vector.
    :return: Cosine similarity score.
    """
    # print(type(vec1), type(vec2))
    if not np.any(vec1) or not np.any(vec2):
        return 0.0  # Avoid division by zero
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_most_relevant_files(
    embeddings: Dict[str, np.ndarray],
    prompt_embedding: np.ndarray,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Retrieves the most relevant files based on cosine similarity.

    :param embeddings: Dictionary with filenames as keys and embedding vectors as values.
    :param prompt_embedding: Embedding vector of the prompt.
    :param top_n: Number of top relevant files to retrieve.
    :return: List of tuples containing filenames and their similarity scores.
    """
    similarities = {}
    for filename, embedding in embeddings.items():
        similarity = compute_cosine_similarity(prompt_embedding, embedding['embedding'])
        similarities[filename] = similarity

    # Sort the files based on similarity scores in descending order
    sorted_files = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_files[:top_n]


def generate_diff(files_in: dict, rag_model: dict, prompt: str, regenerate: bool, diff="") -> str:
    """
    Generates a unified diff based on the current codebase and the provided prompt.

    :param files_dict: Dictionary with filenames as keys and file contents as values.
    :param prompt: Description of changes to make to the codebase.
    :return: Unified diff as a string.
    """
    # Load environment variables from .env
    load_dotenv()

    # Initialize OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    files_dict = files_in
    
    embed_prompt = embed_text(prompt)
    
    rag_outputs = get_most_relevant_files(rag_model, embed_prompt)
    
    files_dict = {file: files_dict[file] for file, _ in rag_outputs}
    
    # print("FILES IN: ", files_dict)
    
    # Validate inputs
    if not isinstance(files_dict, dict):
        raise ValueError("files_dict must be a dictionary with filenames as keys and file contents as values.")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string describing the changes to make.")

    # Construct the files section for the prompt
    files_text = ""
    for filename, content in files_dict.items():
        files_text += f"Filename: {filename}\nContent:\n{content}\n\n"

    # Define the system prompt
    system_prompt = (
        "You are an assistant that generates unified diffs for Git repositories based on user prompts."
        " Below is the current state of the repository's files."
    )
    
    if regenerate:
        system_prompt = (
            "You are an assistant that generates unified diffs for Git repositories based on user prompts."
            " Below is the current state of the repository's files."
            " The previous diff was invalid. Please generate a new diff."
            f" Previous Diff: {diff}\n\n"
        )

    # Define the user prompt
    user_prompt = (
        f"{files_text}"
        f"User Prompt: {prompt}\n\n"
        f"Generate a unified diff that represents the changes needed to accomplish the prompt."
    )

    try:
        # Make the API call to OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,  # Adjust based on your needs
            temperature=0.2,   # Lower temperature for more deterministic output
        )

        # Extract the diff from the response
        diff = response.choices[0].message.content
        return (files_text, prompt, diff)
    except Exception as e:
        # Handle other unexpected errors
        print(f"Unexpected error: {e}")
        return ""

def generate_reflection(gitFiles: dict, prompt: str, diff: str):
    system_prompt = (
        "You are an assistant that generates an analysis on a generated git diff based on user prompts."
        " Ensure that the diff is in a valid unified diff format and that it accomplishes the prompt."
        " If the diff is invalid, provide feedback on how to improve it and regenerate a new diff through the function call."
        " Below is the current state of the repository's files, the user's prompt, and the generated diff."
    )

    # Define the user prompt
    user_prompt = (
        f"Repository Files: {gitFiles}\n\n"
        f"User Prompt: {prompt}\n\n"
        f"Generated Diff: {diff}\n\n"
        f"Generate a unified diff that represents the changes needed to accomplish the prompt."
    )
    
    func_call = {
        "name": "regenerate_diff",
        "description": "Regenerate the diff based on the user prompt.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": ["valid"],
            "additionalProperties": False,
        }
    }

    try:
        # Make the API call to OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,  # Adjust based on your needs
            temperature=0.2,   # Lower temperature for more deterministic output
            tools=[{"type": "function",
                    "function": func_call}]
        )

        # Extract the diff from the response
        reflection = response.choices[0].message.content
        func_args = response.choices[0].message.tool_calls if 'tool_calls' in response.choices[0].message else None
        
        if func_args != None:
            return (reflection, True)
        
        return (reflection, False)
    except Exception as e:
        # Handle other unexpected errors
        print(f"Unexpected error: {e}")
        return ""

def prompt_to_diff(repoUrl: str, prompt: str):
    regenerate = True
    diff = ""
    files_dict = get_repo_files(repoUrl)
    rag_model = analyze_codebase(files_dict)
    while regenerate:
        files, prompt, diff = generate_diff(files_dict, rag_model, prompt, not regenerate, diff)
        reflection, regenerate = generate_reflection(files, prompt, diff)
    return reflection, diff
    

# Testing
# if __name__ == "__main__":
#     repo_url = "https://github.com/JasYi/Formulate"  # Replace with your target repo
#     prompt = "change all openai calls to use anthropic claude"
    
#     regenerate = True
#     diff = ""
#     while regenerate:
#         files, prompt, diff = generate_diff(repo_url, prompt, not regenerate, diff)
#         reflection, regenerate = generate_reflection(files, prompt, diff)
#     print(reflection)
#     print("DIFF: \n\n:" + diff)
