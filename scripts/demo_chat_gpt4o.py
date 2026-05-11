from utils_chat import chat_gpt4o
import sys

# Image insertion format:
# <img src='{file}'>

if __name__ == "__main__":
    # PROMPT_PATH = "/Users/mansionchieng/Workspaces/vlm_workspace/workspace/agent/icl_infer_prompt.txt"
    PROMPT_PATH = sys.argv[1]
    with open(PROMPT_PATH, "r") as f:
        prompt = f.read()
    
    response, _ = chat_gpt4o(prompt)
    print(response)
