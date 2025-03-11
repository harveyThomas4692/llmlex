"""
Module for symbolic regression using Kolmogorov-Arnold Networks (KANs).

This module provides functionality to train KAN models on data,
extract symbolic expressions from the trained models,
simplify these expressions, and fit them to data.
"""

import numpy as np
import torch
import sympy
from sympy import symbols, simplify
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import copy
from kan import KAN, create_dataset

import sympy as sp
from sympy import symbols, simplify
import sympy as sp
import LLMSR.llmSR as llmSR
from sympy import sin, cos, exp, log, sqrt, sinh, cosh, tanh

# Mapping dictionaries for function conversion
numpy_to_sympy = {
    'sin': sympy.sin,
    'cos': sympy.cos,
    'tan': sympy.tan,
    'exp': sympy.exp,
    'log': sympy.log,
    'sqrt': sympy.sqrt,
    'abs': sympy.Abs,
    'arcsin': sympy.asin,
    'arccos': sympy.acos,
    'arctan': sympy.atan,
    'sinh': sympy.sinh,
    'cosh': sympy.cosh,
    'tanh': sympy.tanh,
    'arcsinh': sympy.asinh,
    'arccosh': sympy.acosh,
    'arctanh': sympy.atanh,
    'max': sympy.Max,
    'min': sympy.Min,
    'maximum': sympy.Max,
    'minimum': sympy.Min,
    'abs': sympy.Abs,
    'heaviside': sympy.Heaviside
}

sympy_to_numpy = {
    'sin': "np.sin",
    'cos': "np.cos",
    'tan': "np.tan",
    'exp': "np.exp",
    'log': "np.log",
    'sqrt': "np.sqrt",
    'Abs': "np.abs",
    'asin': "np.arcsin",
    'acos': "np.arccos",
    'atan': "np.arctan",
    'sinh': "np.sinh",
    'cosh': "np.cosh",
    'tanh': "np.tanh",
    'asinh': "np.arcsinh",
    'acosh': "np.arccosh",
    'atanh': "np.arctanh",
    'pi': "np.pi",
    'Max': "np.max",
    'Min': "np.min",
    'Max': "np.max",
    'Min': "np.min",
    'Abs': "np.abs",
    'Heaviside': "np.heaviside"
}


def create_kan_model(width, grid, k, seed=17, symbolic_enabled=False, device='cpu'):
    """
    Create a KAN model with specified parameters.
    
    Args:
        width: List specifying the network architecture (e.g., [1,4,1])
        grid: Grid size for KAN
        k: Number of basis functions
        seed: Random seed for reproducibility
        symbolic_enabled: Whether to enable symbolic features
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        A configured KAN model instance
    """
    try:
        from kan import KAN
    except ImportError:
        raise ImportError("KAN package is required. Please install it first.")
    
    torch.manual_seed(seed)
    model = KAN(width=width, grid=grid, k=k, device=device, symbolic_enabled=symbolic_enabled)
    return model




def subst_params(a, p):
    """
    Substitute numeric values for parameters in expressions.
    
    Args:
        a: Expression with parameter placeholders
        p: List of parameter values
        
    Returns:
    
        Expression with parameter values substituted
    """
    for i in range(len(p)):
        a = a.replace(f'params[{i}]', f'{p[i]:.4f}')
    return a

# def simplify_expression(formula, N=3):
#     """
#     Simplify a mathematical expression using sympy.
    
#     Args:
#         formula: The formula to simplify
#         N: Number of simplification iterations
        
#     Returns:
#         Simplified expression
#     """
#     x = symbols('x')
    
#     try:
#         # Convert string formula to sympy expression
#         if isinstance(formula, str):
#             expr = sympy.sympify(formula)
#         else:
#             expr = formula
            
#         # Apply simplification multiple times
#         for _ in range(N):
#             expr = simplify(expr)
            
#         return expr
#     except Exception as e:
#         print(f"Error simplifying expression: {e}, formula was: {formula}")
#         return formula

def simplify_expression(formula, N=3):
    """
    Simplify a mathematical expression using sympy.
    
    Args:
        formula: The formula to simplify
        N: Number of simplification iterations
    """
    # Define symbolic variables and functions
    variables = symbols(f'x0:{N+1}')
    used_functions = {name: numpy_to_sympy[name] for name in numpy_to_sympy if f'{name}' in formula}
    safe_dict = {f'x{i}': variables[i] for i in range(N+1)}
    safe_dict.update(used_functions)  # Add only used symbolic functions
    try:
        expr = simplify(eval(formula.replace("np.", ""), safe_dict))  # Remove "np." prefix for SymPy functions
    except Exception as e:
        print(f"Error simplifying expression: {e}, formula was: {formula}")
        raise e
    return str(expr)



def replace_floats_with_params(expr_str):
    # Extract float numbers using regex
    float_pattern = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?')
    float_matches = list(set(float_pattern.findall(expr_str)))
    
    # Sort floats by length to avoid partial replacements
    float_matches.sort(key=len, reverse=True)
    
    # Create symbolic parameters
    params = sp.symbols(f'param:{len(float_matches)}')
    param_values = [float(num) for num in float_matches]
    
    # Replace numeric values in the expression
    for i, num in enumerate(float_matches):
        expr_str = expr_str.replace(num, f'params[{i}]')
    
    return expr_str, param_values

def fit_curve(x, y, curve, params_initial):
    params_opt, _ = curve_fit(curve, x, y, p0=params_initial)
    residuals = y - curve(x, *params_opt)
    chi_squared = np.mean((residuals ** 2) / (np.square(curve(x, *params_opt))+1e-6))
    return params_opt, chi_squared 

def call_model_simplify(client, ranges, expr, gpt_model="openai/gpt-4o", system_prompt=None):
    """Call LLM to simplify a mathematical expression within specified ranges."""
    if system_prompt is None:
        system_prompt = (
            "You are a mathematical simplification expert. Simplify the given function over the specified interval."
            "\n\nConsider these simplification strategies:"
            "\n- Taylor expansion of terms that are small in this interval"
            "\n- Removing negligible terms"
            "\n- Recognizing polynomial patterns as Taylor series terms"
            "\n- Combining like terms and factoring when possible"
            "\n\nYour response must follow this format exactly:"
            "\n```simplified_expression\n[your simplified expression here]\n```"
            "\nOnly include the simplified expression inside the delimiters, nothing else. Do not include the square brackets used in the format specification, any other text, or placeholders."
            "\nThe simplified expression should be a valid mathematical expression that can be evaluated."
        )
    
    prompt = (
        f"Please simplify this mathematical expression:\n\n"
        f"Function: {expr}\n"
        f"Valid interval: {ranges}\n\n"
        f"Return only the simplified expression enclosed in the specified format."
    )
    
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
    )

    out = response.choices[0].message.content
    
    # Extract the expression from between the delimiters
    pattern = r"```simplified_expression\n(.*?)\n```"
    match = re.search(pattern, out, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # Fallback if the model didn't use the requested format
        return out.strip()

def sort_symb_expr(symb_expr):
    """
    Sort symbolic expressions by score.
    
    Args:
        symb_expr: Dictionary of symbolic expressions
    
    Returns:
        Sorted dictionary of symbolic expressions
    """
    symb_expr_sorted = {}
    # build disctionary with all expressions ordered by score
    for kan_conn, sub_res in symb_expr.items():
        if sub_res is None:
            print(f"Could not fit a function for connection {kan_conn}")
            continue
        ordered_elements = sorted([item for sublist in sub_res for item in sublist], key=lambda item: -item['score'])
        symb_expr_sorted[kan_conn] = ordered_elements
        print(f"Approximation for {kan_conn}: {ordered_elements[0]['ansatz'].strip()}")
        print(f"Parameters are {np.round(ordered_elements[0]['params'], 1)}")
    return symb_expr_sorted

def build_expression_tree(model, symb_expr_sorted, top_k=3):
    """
    Build an expression tree from KAN model connections and extract best candidate expressions.
    
    The expression tree is defined over L layers with N nodes per layer, where the expression for node (l, n)
    is the sum over its incoming edges, i.e.:
        node(l, n) = Σ₍c₎ expr(l, c, n)
    The KAN convention for the triple (l, c, n) is: (layer, coming from node in l-1, going to node in layer l).
    
    Args:
        model: Trained pruned KAN model with attributes 'width_in' and 'width_out'.
        symb_expr_sorted: Dictionary mapping connection tuples (layer, from, to) to a list of candidate dicts.
        top_k: Number of candidate expressions to retain per connection.
        
    Returns:
        A dictionary with:
          - "edge_dict": mapping from connection to best candidate expression.
          - "top_k_edge_dicts": list of top-k candidate expressions per connection.
          - "node_tree": intermediate expression tree (each node's expression with placeholders).
          - "full_expressions": list of final simplified full expressions for each output node.
    """
    # Process each connection to select the best candidate expression.
    edge_dict = {}
    top_k_edge_dicts = {}
    
    for kan_conn, sub_res in symb_expr_sorted.items():
        best_expr = None
        best_score = -np.inf
        top_k_candidates = []
        
        if sub_res is None:
            print(f"Could not fit a function for connection {kan_conn}")
            continue
            
        for candidate in sub_res[:top_k]:  # Only consider top_k candidates
            # Clean up the ansatz string.
            ansatz = candidate['ansatz'].replace('*x', ' * x') \
                                        .replace('(x)', '(1. * x)') \
                                        .replace('-x', '-1. * x').strip()
            if "lambda" in ansatz:
                continue  # Skip lambda functions.
                
            score = candidate['score']
            expr = subst_params(ansatz, candidate['params'])
            
            # Add to top-k list
            top_k_candidates.append({
                'expression': expr,
                'score': score,
                'ansatz': ansatz,
                'params': candidate['params']
            })
            
            # Update best expression
            if score > best_score:
                best_score = score
                best_expr = expr
                
        edge_dict[kan_conn] = best_expr
        top_k_edge_dicts[kan_conn] = top_k_candidates
        
        if best_expr is not None:
            print(f"KAN Connection: {kan_conn}, Best Expression: {best_expr}, Score: {best_score:.5f}")
        else:
            print(f"Could not fit a function for connection {kan_conn}")
    
    # Build node tree: each node (l, n) is the sum over incoming edges.
    node_tree = {}
    pruned_model = model  # Assuming the provided model is already pruned.
    for l in range(len(pruned_model.width_in) - 1):
        for n in range(pruned_model.width_in[l+1]):
            node_tree[(l, n)] = " + ".join([
                edge_dict.get((l, c, n), "").replace(' x', f' x[{l-1},{c}]' if l > 0 else f' x{c}')
                for c in range(pruned_model.width_out[l])
                if edge_dict.get((l, c, n), "") != ""
            ])
    for k, v in node_tree.items():
        node_tree[k] = v.replace('+ -', '- ')
    
    # Build full expression, prepopulate with output nodes.
    full_expressions = []
    for o in range(pruned_model.width_in[-1]):
        res = node_tree[(len(pruned_model.width_in) - 2, o)]
        # Traverse down lower layers.
        for l in list(range(len(pruned_model.width_in) - 2))[::-1]:
            n_count = len([x for x in node_tree.keys() if x[0] == l])
            for n in range(n_count):
                res = res.replace(f'x[{l},{n}]', f'({node_tree[(l, n)]})')
        full_expressions.append(simplify_expression(res, pruned_model.width_in[0] - 1))
    
    return {
        "edge_dict": edge_dict,
        "top_k_edge_dicts": top_k_edge_dicts,
        "node_tree": node_tree,
        "full_expressions": full_expressions
    }

def optimize_expression(client, node_tree, gpt_model, x_data, y_data, custom_system_prompt=None, original_f = None, KAN_model = None):
    """
    Optimize and simplify the final expressions.
    
    For each output in the full expression, the function:
      - Prunes and simplifies the expression (replacing floats with parameters and rounding coefficients).
      - Refits the parameters via curve fitting.
      - Plots comparisons with the raw and refitted expressions.
      - Uses an LLM call to further simplify and refit the expression.
    
    Args:
        client: OpenAI client or a compatible client.
        node_tree: Dictionary containing the expression tree and 'full_expression' list.
        gpt_model: Model to use for LLM simplification.
        x_data: x data points (numpy array).
        y_data: y data points (numpy array).
        custom_system_prompt: (Optional) Custom system prompt for LLM.
        
    Returns:
        Dictionary with key 'final_expressions' containing the final refined expressions.
    """
    # Use full expressions if available; otherwise default to the output from the final layer.
    if "full_expressions" in node_tree:
        full_expressions = node_tree["full_expressions"]
    else:
        raise ValueError("No full expression found in node_tree")
    
    # if custom_system_prompt is None:
    #     custom_system_prompt = """
    #     You are a physics assistant. Your task is to simplify mathematical 
    #     expressions as much as possible while preserving their meaning.
    #     """
    
    ranges = (float(np.min(x_data)), float(np.max(x_data)))
    final_expressions = []
    simplified_expressions = []
    chi_squared_finals = []
    chi_squared_simplified = []
    best_chi_squared = float('inf')
    best_expression = None
    best_expression_index = None
    fig, ax = plt.subplots()
    
    for i, expr in enumerate(full_expressions):
        print("###################################################")
        print(f"Simplifying output {i}")
        print("KAN expression (raw):\n", expr)
        f_fitted = lambda x0: eval(expr)
        xs = np.arange(min(x_data), max(x_data), (max(x_data)-min(x_data))/100)
        try:
            try:
                ax.plot(xs, [original_f(torch.tensor(x)) for x in xs], label="Original")
            except TypeError:
                ax.plot(xs, [original_f(x) for x in xs], label="Original")
        except Exception as e:
            print(f"Original function 'f' not defined; skipping original plot {e}")
        ax.plot(xs, [f_fitted(x) for x in xs], label="KANSR (raw)")
        ax.legend()
    
        # Prune and simplify: replace floats with parameters (round coefficients to 4 digits) then simplify.
        expr = simplify_expression(subst_params(*replace_floats_with_params(expr)),
                                   KAN_model.width_in[0] - 1)
        simplified_expressions.append(expr)
        print("KAN expression (simplified):\n", expr)
    
        # Refit parameters.
        curve_ansatz_str, params_initial = replace_floats_with_params(expr)
        for k, v in sympy_to_numpy.items():
            curve_ansatz_str = curve_ansatz_str.replace(k, v)

        print("replaced", curve_ansatz_str)
        
        curve_ansatz = "lambda x0, *params: " + curve_ansatz_str
        curve = eval(curve_ansatz, {"np": np})
        try:
            params_opt, chi_squared = fit_curve(x_data, y_data,
                                                curve, params_initial)
            print(f"Refitting: {curve_ansatz_str} - so after simplification gave a chi^2 of {chi_squared:.4e}")
            chi_squared_simplified.append(chi_squared)
            
            # Track best chi-squared
            if chi_squared < best_chi_squared:
                best_chi_squared = chi_squared
                best_expression = simplify_expression(subst_params(curve_ansatz_str, params_opt),
                                                     KAN_model.width_in[0] - 1)
                best_expression_index = i
        except RuntimeError as e:
            print("Proceeding with unoptimized parameters {e}")
            params_opt = params_initial
            chi_squared_simplified.append(float('inf'))
    
        # Prune and simplify the refitted model.
        expr = simplify_expression(subst_params(curve_ansatz_str, params_opt),
                                   KAN_model.width_in[0] - 1)
        print("KAN expression (final):\n", expr)
        final_expressions.append(expr)
    
        # Plot comparison.
        f_fitted = lambda x0: eval(expr)
        ax.plot(xs, [f_fitted(x) for x in xs], label="KANSR (refitted)")
    
        # Ask LLM to further simplify the result and refit.
        try:
            expr = call_model_simplify(client, ranges, expr, gpt_model, system_prompt=custom_system_prompt)
            logger.info(f"LLM improvement response is: {expr}")
            curve_ansatz_str, params_initial = replace_floats_with_params(expr)
            curve_ansatz = "lambda x0, *params: " + curve_ansatz_str
            curve = eval(curve_ansatz, {"np": np})
            
            params_opt, chi_squared = fit_curve(x_data, y_data,
                                                curve, params_initial)
            chi_squared_finals.append(chi_squared)
            print(f"LLM improvement gave a chi^2 of {chi_squared:.4e}")
            
            # Track best chi-squared
            if chi_squared < best_chi_squared:
                best_chi_squared = chi_squared
                best_expression = simplify_expression(subst_params(curve_ansatz_str, params_opt),
                                                     KAN_model.width_in[0] - 1)
                best_expression_index = i
            expr = simplify_expression(subst_params(curve_ansatz_str, params_opt),
                                       KAN_model.width_in[0] - 1)
            f_fitted = lambda x0: eval(expr)
            ax.plot(xs, [f_fitted(x) for x in xs], label="KANSR (final)")
        except Exception as e:
            print(f"Skipping LLM improvement. {e}")
            chi_squared_finals.append(chi_squared)
    
        ax.legend()
        plt.show()
        print("##################\n# Final formula: #\n##################\n", expr)
    
    result_dict = {
        'raw_expressions': full_expressions,
        'simplified_expressions': simplified_expressions,
        'final_expressions': final_expressions,
        'chi_squared_finals': chi_squared_finals,
        'chi_squared_simplified': chi_squared_simplified,
        'best_expression': best_expression,
        'best_chi_squared': best_chi_squared,
        'best_expression_index': best_expression_index
    }
    
    return best_expression, result_dict

def plot_results(f, ranges, result_dict, title="KAN Symbolic Regression Results"):
    """
    Plot the original function and the approximations.
    
    Args:
        f: Original function
        ranges: Tuple of (min_x, max_x) for the input range
        result_dict: Dictionary with results from optimize_expression
        title: Plot title
        
    Returns:
        matplotlib figure and axes objects
    """
    x = np.linspace(ranges[0], ranges[1], 1000)
    y_true = f(torch.tensor(x).reshape(-1, 1).float()).numpy().flatten()
    
    # Create a function to evaluate expressions
    def eval_expr(expr_str, x_vals):
        y_vals = []
        for x_val in x_vals:
            x0 = x_val  # Variable name used in expressions
            y_vals.append(eval(expr_str))
        return np.array(y_vals)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the true function
    ax.plot(x, y_true, 'k-', label='True Function', linewidth=2)
    
    # Get the best expression
    best_expr = result_dict['best_expression']
    best_chi_squared = result_dict['best_chi_squared']
    
    # Try to plot the best expression
    try:
        y_best = eval_expr(best_expr, x)
        ax.plot(x, y_best, 'r--', label=f'Best Expression (χ²={best_chi_squared:.5e})', linewidth=2)
    except Exception as e:
        print(f"Error plotting best expression: {e}")
    
    # Try to plot the simplified expression
    try:
        best_idx = result_dict['best_expression_index']
        simplified_expr = result_dict['simplified_expressions'][best_idx]
        chi_squared = result_dict['chi_squared_simplified'][best_idx]
        
        y_simplified = eval_expr(simplified_expr, x)
        ax.plot(x, y_simplified, 'g-.', label=f'Simplified (χ²={chi_squared:.5e})', linewidth=2)
    except Exception as e:
        print(f"Error plotting simplified expression: {e}")
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def run_complete_pipeline(client, f, ranges=(-np.pi, np.pi), width=[1,4,1], grid=7, k=3, 
                         train_steps=50, generations=3, gpt_model="openai/gpt-4o", device='cpu',
                         node_th=0.2, edge_th=0.2, custom_system_prompt_for_second_simplification=None, optimizer="LBFGS", population=10, temperature=0.1, exit_condition=1e-3, verbose=0, use_async=True, plot_fit=True, plot_parents=False):
    """
    Run the complete KAN symbolic regression pipeline.
    
    Args:
        client: OpenAI client or compatible client
        f: Target function to approximate
        x_range: Tuple of (min_x, max_x) for the input range
        width: List specifying the network architecture
        grid: Grid size for KAN
        k: Number of basis functions
        train_steps: Number of training steps
        generations: Number of generations in the genetic algorithm
        gpt_model: Model to use for simplification
        device: Device to use ('cpu' or 'cuda')
        node_th: Node threshold for pruning
        edge_th: Edge threshold for pruning
        custom_system_prompt_for_second_simplification: Custom system prompt for LLM second simplification
        
    Returns:
        Dictionary with complete results including models and expressions:
        - 'trained_model': Trained KAN model
        - 'pruned_model': Pruned KAN model
        - 'train_loss': Training loss
        - 'symbolic_expressions': Symbolic expressions
        - 'node_tree': Node tree
        - 'result': Result dictionary
    """
    # 1. Create the model
    model = create_kan_model(width, grid, k, device=device)
    
    # 2. Create the dataset
    dataset = create_dataset(f, n_var=1, ranges=ranges, train_num=10000, test_num=1000, device=device)

           # Train the model
    res = model.fit(dataset, opt=optimizer, steps=train_steps)
    
    # Prune the model
    pruned_model = model.prune(node_th=node_th, edge_th=edge_th)

    print("Trained model:")
    model.plot()
    print("Pruned model:")
    pruned_model.plot()
    train_loss = res['train_loss']
    # 4. Convert to symbolic expressions
    res = llmSR.kan_to_symbolic(pruned_model, client, population=population, generations=generations, temperature=temperature, gpt_model=gpt_model,
                                exit_condition=min(train_loss).item(), verbose=verbose, use_async=use_async, plot_fit=plot_fit, plot_parents=plot_parents)
    symb_expr_sorted = sort_symb_expr(res)
    
    # 5. Build expression tree
    node_data = build_expression_tree(pruned_model, symb_expr_sorted, top_k=3)
    # 6. Optimize expression
    # Convert training data to numpy arrays for optimization
    x_data = dataset['train_input'].cpu().numpy().flatten()
    y_data = dataset['train_label'].cpu().numpy().flatten()
    # Optimize and simplify the expression
    best_expression, result_dict = optimize_expression(
        client, node_data, gpt_model, x_data, y_data, 
        custom_system_prompt=custom_system_prompt_for_second_simplification, original_f=f, KAN_model=pruned_model
    )

    # Print the results
    best_index = result_dict['best_expression_index']
    print(f"best expression: {result_dict['best_expression']}, at index {best_index}, with chi^2 {result_dict['best_chi_squared']}")
    print(f"initially: {result_dict['raw_expressions'][best_index]}")
    print(f"then simplified: {result_dict['simplified_expressions'][best_index]}, chi^2 {result_dict['chi_squared_simplified'][best_index]}")
    print(f"then refitted: {result_dict['final_expressions'][best_index]}, chi^2 {result_dict['chi_squared_finals'][best_index]}")

    # 7. Combine everything into a results dictionary
    return {
        'trained_model': model,
        'pruned_model': pruned_model,
        'train_loss': train_loss,
        'symbolic_expressions': symb_expr_sorted,
        'node_tree': node_data,
        'result': result_dict,
        'dataset': dataset,
        'best_expression': best_expression,
        'best_chi_squared': result_dict['best_chi_squared']
    }
