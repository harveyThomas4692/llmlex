import requests
import openai

def check_key_limit(client):
    '''
    Checks the remaining limit for the provided API key.
    Args:
        client (object): An object containing the API key and base URL.
    Returns:
        int: The remaining limit for the API key if the request is successful.
        str: The error message if the request fails.
    '''
    headers = {"Authorization": f"Bearer {client.api_key}",}

    response = requests.get(f"{client.base_url}/auth/key", headers=headers)

    if response.status_code == 200:
        # Print the response data
        return response.json()["data"]['limit_remaining']
    else:
        # Print an error message with the status code
        print(f"Request failed with status code {response.status_code}")
        return response.text
    
def get_prompt(function_list = None):
    """
    Generates the user prompt given a list of functions.
    Args:
        function_list (list, optional): A list of strings where each string is a mathematical expression to be used in the lambda functions. 
                                        If None, defaults to ["params[0]"].
    Returns:
        str: A string containing the generated user prompt with lambda function definitions.
    """

    if function_list is None:
        function_list = ["params[0]"]
    
    prompt = "import numpy as np \n"
    for n in range(len(function_list)):
        prompt += f"curve_{n} = lambda x,*params: {function_list[n]} \n"
    prompt += f"curve_{len(function_list)} = lambda x,*params:"

    return prompt

def call_model(client, model, image, prompt, system_prompt=None):
    """
    In initiates a single call of the llm given an image. 
    Args:
        client (object): The client object used to interact with the model.
        model (str): The name or identifier of the model to be used.
        image (str): The image data encoded in base64 format.
        prompt (str): The text prompt provided by the user.
        system_prompt (str, optional): The system prompt to guide the model's response. If none provided it defaults to:
                                        "Give an improved ansatz to the list for the image. Follow on from the users text with no explaining.
                                        Params can be any length. There's some noise in the data, give preference to simpler functions"
    Returns:
        dict: The response from the model, typically containing the generated text or other relevant information.
    """
    if system_prompt is None:
        system_prompt = ("Give an improved ansatz to the list for the image. Follow on from the users text with no explaining."
                         "Params can be any length. There's some noise in the data, give preference to simpler functions.")

    response = client.chat.completions.create(
    model=model,
    messages=[
        { "role": "system", 
         "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
        ],
        max_tokens=4096,
        )
    
    return response