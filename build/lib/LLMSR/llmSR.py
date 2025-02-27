import numpy as np
import re
from scipy.optimize import curve_fit
import requests
import base64
from tqdm import tqdm
import matplotlib.pyplot as plt
import io

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def generate_base64_image(fig, ax, x, y):
    ax.clear()  # Clear previous plot
    ax.plot(x, y, label='data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()

    # Save to buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)  # Reset buffer position
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8") # encode img into buffer
    buffer.close()  # Close buffer
    return base64_image

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
        function_list = ["params[0]"]
    
    prompt = "import numpy as np \n"
    for n in range(len(function_list)):
        prompt += f"curve_{n} = lambda x,*params: {function_list[n]} \n"
    prompt += f"curve_{len(function_list)} = lambda x,*params:"

    return prompt

def call_model(client, model, image, prompt, system_prompt=None):
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
        chi_squared = np.mean((residuals ** 2) / (np.square(curve(x, *params_opt))+1e-6))
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
            # "curve": curve, This wouldn't run for some reason
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
    print("Checking constant function")
    curve = lambda x, *params: params[0] * np.ones(len(x))
    params, _ = curve_fit(curve, x, y, p0=[1])
    residuals = y - params[0]*np.ones(len(x))
    chi_squared = np.mean((residuals ** 2) / (np.square(curve(x, *params))+1e-6))

    if chi_squared <= exit_condition:
        print("Constant function is good fit.")
        print("Score: ", -chi_squared)
        print("Constant: ", params)
        populations.append([{
            "params": params,
            "score": -chi_squared,
            "ansatz": "lambda x,*params: params[0] * np.ones(len(x))",
            "Num_params": 0,
            "response": None,
            "prompt": None,
            "function_list": None
        }])
        return populations
    print("Constant function is not a good fit.")
    print("Score: ", -chi_squared)
    print("Constant: ", params)

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

def generate_nested_functions(layer_connections):
    """
    Generates a nested function expression for a symbolic computational graph.

    Parameters:
        layer_connections (dict): A dictionary where keys are layer indices (1 to L),
                                  and values are lists of nodes in that layer, 
                                  each containing a list of indices from the previous layer.

    Returns:
        str: Nested function string.
    """
    L = max(layer_connections.keys())
    def construct_layer(l, node):
        """Recursively constructs the function for a given node at layer l."""
        if l == 0:
            # Base case: Input layer directly maps to x_i
            return f"f_{{0,{node}}}(x_{node})"
        else:
            # Construct sum of incoming connections
            inputs = " + ".join([f"f_{{{l},{node}}}({construct_layer(l-1, prev)})" 
                                 for prev in layer_connections[l].get(node, [])])
            return f"({inputs})" if inputs else f"f_{{{l},{node}}}()"  # Handle empty case

    # Construct output as a list
    return [construct_layer(L, node) for node in layer_connections[L]]

def to_symbolic(model, client, population=10, generations=3, temperature=0.1, gpt_model="openai/gpt-4o-mini", exit_condition=1e-3):
    res, res_fcts = 'Sin', {}
    layer_connections = {0: {i: [] for i in range(model.width_in[0])}}
    for l in range(len(model.width_in) - 1):
        layer_connections[l] = {i: list(range(model.width_out[l-1])) if l > 0 else []  for i in range(model.width_in[l])}
    symb_formula = generate_nested_functions(layer_connections)
    
    for l in range(len(model.width_in) - 1):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l + 1]):
                if (model.symbolic_fun[l].mask[j, i] > 0. and model.act_fun[l].mask[i][j] == 0.):
                    print(f'skipping ({l},{i},{j}) since already symbolic')
                    symb_formula = [s.replace(f'f_{{{l},{i},{j}}}', 'TODO') for s in symb_formula]
                elif (model.symbolic_fun[l].mask[j, i] == 0. and model.act_fun[l].mask[i][j] == 0.):
                    model.fix_symbolic(l, i, j, '0', verbose=verbose > 1, log_history=False)
                    print(f'fixing ({l},{i},{j}) with 0')
                    symb_formula = [s.replace(f'f_{{{l},{i},{j}}}', '0') for s in symb_formula]
                    res_fcts[(l, i, j)] = None
                else:
                    x_min, x_max, y_min, y_max = model.get_range(l, i, j, verbose=False)
                    x, y = model.acts[l][:, i].cpu().detach().numpy(), model.spline_postacts[l][:, j, i].cpu().detach().numpy()
                    ordered_in = np.argsort(x)
                    x, y = x[ordered_in], y[ordered_in]
                    fig, ax = plt.subplots()
                    plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                    plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                    base64_image = generate_base64_image(fig, ax, x, y)
                    print((l,i,j))
                    plt.show()
                    mask = model.act_fun[l].mask
                    try:
                        res = run_genetic(client, base64_image, x, y, population, generations, temperature=temperature, model=gpt_model, system_prompt=None, elite=False, exit_condition=exit_condition)
                        res_fcts[(l,i,j)] = res
                        # symb_formula = [s.replace(f'f_{{{l},{i}}}', res) for s in symb_formula]
                    except Exception as e:
                        print(e)
                        res_fcts[(l,i,j)] = None
    ax.clear()
    plt.close()
    return res_fcts, symb_formula