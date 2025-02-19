import numpy as np
import re
from scipy.optimize import curve_fit
import requests
import base64
from tqdm import tqdm

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def check_key_limit(client):
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
    if function_list is None:
        function_list = ["x* params[0] + params[1]"]
    
    prompt = "import numpy as np \n"
    for n in range(len(function_list)):
        prompt += f"curve_{n} = lambda x,*params: {function_list[n]} \n"
    prompt += f"curve_{len(function_list)} = lambda x,*params:"

    return prompt

def call_model(client, model, image, prompt, system_prompt=None):
    if system_prompt is None:
        system_prompt = "Give an improved ansatz to the list for the image. Follow on from the users text with no explaining. Params can be any length."

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

def extract_ansatz(response):
    text = response.choices[0].message.content
    ansatz = text.split('\n')[1]
    if ansatz.startswith("_curve") or ansatz.startswith("curve"):
        ansatz = ansatz.split(":", 1)[1]

    # Extract the values of params from the string
    params_values = re.findall(r'params\[(\d+)\]', ansatz)

    # Convert the extracted values to integers
    params_values = list(map(int, params_values))

    # Find the largest entry
    largest_entry = max(params_values) + 1
    return ansatz, largest_entry

def fun_convert(ansatz):
    ansatz = "lambda x,*params: " + ansatz
    curve = eval(ansatz)
    return curve

def fit_curve(x, y, curve, largest_entry):
    try:
        params_initial = np.ones(largest_entry)
        params_opt, _ = curve_fit(curve, x, y, p0=params_initial)
        residuals = y - curve(x, *params_opt)
        chi_squared = np.mean((residuals ** 2) / (np.abs(curve(x, *params_opt))+1e-6))
        return params_opt, chi_squared
    except Exception as e:
        #print(e)
        return np.array(params_initial), np.inf
    
def single_call(client, img, x, y, model="openai/gpt-4o-mini",function_list=None, system_prompt=None):
    try:
        prompt = get_prompt(function_list)
        resp = call_model(client, model, img, prompt, system_prompt=system_prompt)
        out = extract_ansatz(resp)
        curve = fun_convert(out[0])
        params, uncert = fit_curve(x, y, curve, out[1])

        return {
            "params": params,
            "score": -uncert,
            "ansatz": out[0],
            "Num_params": out[1],
            "curve": curve,
            "response": resp,
            "prompt": prompt,
            "function_list": function_list
        }
    except Exception as e:
        return None

def run_genetic(client, base64_image, x, y, population_size,num_of_generations,
                 temperature=1., model="openai/gpt-4o-mini", exit_condition=1e-5,system_prompt=None, elite=False):
    population = []
    populations = []
    print("Generating Initial population population")
    for i in tqdm(range(population_size)):
        good = False
        while not good:
            result = single_call(client, base64_image, x, y, model=model, system_prompt=system_prompt)
            if result is not None:
                population.append(result)
                good = True

    population.sort(key=lambda x: x['score'])
    populations.append(population)
    best_pop = population[-1]
    print("Best score: ", best_pop['score'])
    print("Best ansatz: ", best_pop['ansatz'])
    print("Best params: ", best_pop['params'])
    if best_pop['score'] > -exit_condition:
        print("Exit condition met.")
        return populations

    for generation in range(num_of_generations-1):
        print(f"Generation: {generation+1}")
        scores = np.array([ind['score'] for ind in population])
        finite_scores = scores[np.isfinite(scores)]
        normalized_scores = (scores - np.min(finite_scores)) / (np.max(finite_scores) - np.min(finite_scores) + 1e-6)
        exp_scores = np.exp((normalized_scores - np.max(normalized_scores))/temperature)
        probs = exp_scores / np.sum(exp_scores)

        selected_population = [np.random.choice(populations[-1], size=2,
                                                 p=probs, replace=True) for _ in range(population_size)]

        func_lists = [[pops[0]['ansatz'],pops[1]['ansatz']] for pops in selected_population]
        population = []
        if elite:
            population.append(best_pop)
        for funcs in tqdm(range(population_size)):
            good = False
            while not good:
                result = single_call(client, base64_image, x, y, model=model,
                                      function_list=func_lists[funcs], system_prompt=system_prompt)
                if result is not None:
                    population.append(result)
                    good = True
        
        population.sort(key=lambda x: x['score'])
        best_pop = population[-1]
        populations.append(population)
        print("Best score: ", best_pop['score'])
        print("Best ansatz: ", best_pop['ansatz'])
        print("Best params: ", best_pop['params'])
        if best_pop['score'] > -exit_condition:
            print("Exit condition met.")
            return populations
        
    return populations