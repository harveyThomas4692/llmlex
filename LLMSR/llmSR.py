import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt
from LLMSR.images import generate_base64_image
from LLMSR.llm import get_prompt, call_model
from LLMSR.response import extract_ansatz, fun_convert

import LLMSR.fit as fit
    
def single_call(client, img, x, y, model="openai/gpt-4o-mini",function_list=None, system_prompt=None):
    """
    Executes a single call of to a specified llm-model with given parameters and processes the response.
    Args:
        client: The openai client object used to interact with the llm.
        img: The base64 encoded image to use for the model.
        x: The x-values for curve fitting.
        y: The y-values for curve fitting.
        model (str, optional): The model identifier to be used. Defaults to "openai/gpt-4o-mini".
        function_list (list, optional): A list of functions to be included in the prompt. Defaults to None.
        system_prompt (str, optional): A system-level prompt to guide the model's behavior. Defaults to None.
    Returns:
        dict: A dictionary containing the following keys:
            - "params": The parameters resulting from the curve fitting.
            - "score": The score of the curve fitting.
            - "ansatz": The ansatz extracted from the model's response.
            - "Num_params": The number of parameters in the ansatz.
            - "response": The raw response from the model.
            - "prompt": The prompt used in the model call.
            - "function_list": The list of functions included in the prompt.
        None: If an exception occurs during the process.
    """

    try:
        prompt = get_prompt(function_list)
        resp = call_model(client, model, img, prompt, system_prompt=system_prompt)
        out = extract_ansatz(resp)
        curve = fun_convert(out[0])
        params, score = fit.fit_curve(x, y, curve, out[1])

        return {
            "params": params,
            "score": -score,
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
                temperature=1., model="openai/gpt-4o-mini", exit_condition=1e-5,system_prompt=None, elite=False, for_kan=False):
    """
        Run a genetic algorithm to fit a model to the given data.
        Parameters:
            client (object): The client object to use for API calls.
            base64_image (str): The base64 encoded image to use for the model.
            x (array-like): The input data for the model.
            y (array-like): The target data for the model.
            population_size (int): The size of the population for the genetic algorithm.
            num_of_generations (int): The number of generations to run the genetic algorithm.
            temperature (float, optional): The temperature parameter for the selection process. Default is 1.
            model (str, optional): The model to use for the API calls. Default is "openai/gpt-4o-mini".
            exit_condition (float, optional): The exit condition for the genetic algorithm. Default is 1e-5.
            system_prompt (str, optional): The system prompt to use for the API calls. Default is None.
            elite (bool, optional): Whether to use elitism in the genetic algorithm. Default is False.
        Returns:
            list: A list of populations, where each population is a list of individuals.
        """
    
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
            "ansatz": "lambda x,*params: params[0] * np.ones(len(x))" if for_kan else "params[0]",
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

    for p in population:
        if np.isnan(np.sum(p['score'])):
            p['score'] = -1e8
            
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
        exp_scores = np.nan_to_num(exp_scores, nan=0.0)
        if np.all(exp_scores == 0):
            exp_scores = np.ones_like(exp_scores)
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


def kan_to_symbolic(model, client, population=10, generations=3, temperature=0.1, gpt_model="openai/gpt-4o-mini", exit_condition=1e-3):
    """
    Converts a given kan model symbolic representations using llmsr.
    Parameters:
        model (object): The kan model.
        client (object): The openai client object used to access the llm.
        population (int, optional): The population size for the genetic algorithm. Default is 10.
        generations (int, optional): The number of generations for the genetic algorithm. Default is 3.
        temperature (float, optional): The temperature parameter for the genetic algorithm. Default is 0.1.
        gpt_model (str, optional): The GPT model to use for generating symbolic functions. Default is "openai/gpt-4o-mini".
        exit_condition (float, optional): The exit condition for the genetic algorithm. Default is 1e-3.
    Returns:
        - res_fcts (dict): A dictionary mapping layer, input, and output indices to their corresponding symbolic functions.
    """

    res, res_fcts = 'Sin', {}
    layer_connections = {0: {i: [] for i in range(model.width_in[0])}}
    for l in range(len(model.width_in) - 1):
        layer_connections[l] = {i: list(range(model.width_out[l-1])) if l > 0 else []  for i in range(model.width_in[l])}
    
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
                        res = run_genetic(client, base64_image, x, y, population, generations, temperature=temperature, model=gpt_model, system_prompt=None, elite=False, exit_condition=exit_condition, for_kan=True)
                        res_fcts[(l,i,j)] = res
                    except Exception as e:
                        print(e)
                        res_fcts[(l,i,j)] = res
    ax.clear()
    plt.close()
    return res_fcts