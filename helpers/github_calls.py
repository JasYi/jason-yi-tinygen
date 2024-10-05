from github import Github
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

load_dotenv()

# Replace with your GitHub Personal Access Token if accessing private repositories
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Optional for public repos

# Given 
def is_text_file_by_decoding(content: bytes) -> bool:
    try:
        content.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

def get_repo_owner_and_name(repo_url):
    """
    Extracts the owner and repository name from the GitHub URL.
    """
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub repository URL.")
    owner, repo = path_parts[0], path_parts[1].replace('.git', '')
    return owner, repo

def list_all_files_pygithub(repo_url):
    """
    Lists all files in the repository recursively using PyGithub.
    """
    owner, repo_name = get_repo_owner_and_name(repo_url)
    g = Github(GITHUB_TOKEN) if GITHUB_TOKEN else Github()
    try:
        repo = g.get_repo(f"{owner}/{repo_name}")
    except Exception as e:
        raise Exception(f"Failed to access repository: {e}")
    
    contents = repo.get_contents("")
    files = []
    queue = contents[:]
    
    while queue:
        content = queue.pop(0)
        if content.type == "dir":
            try:
                queue.extend(repo.get_contents(content.path))
            except Exception as e:
                print(f"Error accessing directory {content.path}: {e}")
        elif content.type == "file":
            files.append(content.path)
    
    return files

def read_all_files_pygithub(repo_url, file_names):
    """
    Reads the content of all files in the repository recursively using PyGithub.
    Returns a dictionary with file paths as keys and file contents as values.
    """
    owner, repo_name = get_repo_owner_and_name(repo_url)
    g = Github(GITHUB_TOKEN) if GITHUB_TOKEN else Github()
    try:
        repo = g.get_repo(f"{owner}/{repo_name}")
    except Exception as e:
        raise Exception(f"Failed to access repository: {e}")
    
    files_contents = {}
    queue = file_names[:]
    
    while queue:
        content = queue.pop(0)
        file_obj = repo.get_contents(content)
        print("Reading file:", content)
        if file_obj.type == "file":
            try:
                if is_text_file_by_decoding(file_obj.decoded_content):
                    file_content = file_obj.decoded_content.decode('utf-8')
                    files_contents[file_obj.path] = file_content
                else:
                    print(f"File {file_obj.path} is not readable file.")
            except Exception as e:
                print(f"Error reading file {file_obj.path}: {e}")
    
    return files_contents

def get_repo_files(repo_url):
    """
    Returns a list of all files in the repository and their contents.
    """
    all_files = list_all_files_pygithub(repo_url)
    files_contents = read_all_files_pygithub(repo_url, all_files)
    return files_contents

# Testing
if __name__ == "__main__":
    repo_url = "https://github.com/jayhack/llm.sh"  # Replace with your target repo
    print(get_repo_files(repo_url))
