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
from sympy.printing.numpy import NumPyPrinter
from LLMSR.fit import get_chi_squared, fit_curve_with_guess

# Mapping dictionaries for function conversion
numpy_to_sympy = {
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'exp': sp.exp,
    'log': sp.log,
    'sqrt': sp.sqrt,
    'abs': sp.Abs,
    'arcsin': sp.asin,
    'arccos': sp.acos,
    'arctan': sp.atan,
    'atan2': sp.atan2,
    'atanh': sp.atanh,
    'atan': sp.atan,
    'sinh': sp.sinh,
    'cosh': sp.cosh,
    'tanh': sp.tanh,
    'arcsinh': sp.asinh,
    'arccosh': sp.acosh,
    'arctanh': sp.atanh,
    'max': sp.Max,
    'min': sp.Min,
    'maximum': sp.Max,
    'minimum': sp.Min,
    'heaviside': sp.Heaviside,
    'power': sp.Pow,
    'gamma': sp.gamma,
    'Gamma': sp.gamma,
    'factorial': sp.factorial,
    'Factorial': sp.factorial,
    'erf': sp.erf,
    'Erf': sp.erf,
    'erfc': sp.erfc,
    'Erfc': sp.erfc,
}

# sympy_to_numpy = {
 
#     'sin': "np.sin",
#     'cos': "np.cos",
#     'tan': "np.tan",
#     'exp': "np.exp",
#     'log': "np.log",
#     'sqrt': "np.sqrt",
#     'Abs': "np.abs",
#     'asin': "np.arcsin",
#     'acos': "np.arccos",
#     'atan': "np.arctan",
#     'sinh': "np.sinh",
#     'cosh': "np.cosh",
#     'tanh': "np.tanh",
#     'asinh': "np.arcsinh",
#     'acosh': "np.arccosh",
#     'atanh': "np.arctanh",
#     'pi': "np.pi",
#     'Max': "np.max",
#     'Min': "np.min",
#     'Max': "np.max",
#     'Min': "np.min",
#     'Abs': "np.abs",
#     'Heaviside': "np.heaviside",
#     'Pow': "np.power",
#     'gamma': "np.gamma",
#     'Gamma': "np.gamma",
#     'factorial': "np.math.factorial",
#     'Factorial': "np.math.factorial",
#     'erf': "np.math.erf",
#     'Erf': "np.math.erf",
#     'erfc': "np.math.erfc",
#     'Erfc': "np.math.erfc",
# }


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

def convert_sympy_to_numpy(expr):
    expr_str = NumPyPrinter().doprint(expr)
    
    # First convert any 'numpy.' to 'np.' to standardize
    expr_str = re.sub(r'numpy\.', 'np.', expr_str)
    
    # Find functions that don't have np. prefix
    unknown_functions = re.findall(r'(?<!np\.)\b\w+(?=\()', expr_str)
    
    for func in unknown_functions:
        # Skip functions that are already prefixed or are part of a prefixed function
        if func not in ['np', 'numpy'] and not func.startswith('np.') and not func.startswith('numpy.'):
            # Make sure we're not adding prefix to something that's already part of a prefixed function
            expr_str = re.sub(r'(?<!np\.)\b' + func + r'\b(?=\()', f'np.{func}', expr_str)
    
    return re.sub(r'lambda[^:]*:', '', expr_str)

def simplify_expression(formula, N=10):
    """
    Simplify a mathematical expression using sympy. Converts to sympy expression first.
    
    Args:
        formula: The formula to simplify
    """
    # Define symbolic variables and functions
    variables = symbols(f'x0:{N+1}')
    used_functions = {name: numpy_to_sympy[name] for name in numpy_to_sympy if f'{name}' in formula}
    safe_dict = {f'x{i}': variables[i] for i in range(N+1)}
    safe_dict.update(used_functions)  # Add only used symbolic functions
    safe_dict.update({'sp':sp})
    try:
        formula = formula.replace("np.", "") # Remove "np." prefix for SymPy functions
        for key, value in numpy_to_sympy.items():
            # Replace function names with their sympy equivalents, but avoid replacing if already prefixed with sp.
            formula = re.sub(r'(?<!sp\.)\b' + key + r'\b(?=\()', "sp." + value.__name__, formula, flags=re.IGNORECASE)
        # Find all unknown functions in the formula without 'sp.' prefix
        unknown_functions = re.findall(r'(?<!sp\.)\b\w+\b(?=\()', formula)
        for func in unknown_functions:
            if func not in safe_dict:
                safe_dict[func] = sp.Function(func)
        print("simplifying", formula, "with", safe_dict)
        expr = simplify(eval(formula, safe_dict)) 
    except Exception as e:
        print(f"Error simplifying expression: {e}, formula was: {formula}")
        raise e
    return str(expr)#removes the sp. prefix


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

def call_model_simplify(client, ranges, expr, gpt_model="openai/gpt-4o", system_prompt=None, sympy=True, numpy=False):
    """Call LLM to simplify a mathematical expression within specified ranges.
    
    Args:
        client: OpenAI client or compatible client
        ranges: Tuple of (min, max) values for the interval
        expr: The expression to simplify
        gpt_model: The GPT model to use
        system_prompt: The system prompt to use. If None or 'default', a default prompt is used.
        sympy: Whether to use sympy
        numpy: Whether to use numpy
    """
    if sympy and numpy:
        raise ValueError("Cannot specify both sympy and numpy as True.")
    if system_prompt is None or system_prompt == "default":
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
            f"\nThe simplified expression should be a valid mathematical expression in {'sympy' if sympy else 'numpy'} that can be evaluated."
        )
        print("using default system prompt for LLM simplification")
    else:
        print("using provided system prompt for LLM simplification")
    
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
                                        .replace('-x', '-1. * x').replace('x,',' x ,').replace('(x', '( x').replace('x)', 'x )').replace('/x', '/ x').strip()
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
        full_expressions.append(simplify_expression(res, pruned_model.width_in[0] - 1))# sympy expression
    
    return {
        "edge_dict": edge_dict,
        "top_k_edge_dicts": top_k_edge_dicts,
        "node_tree": node_tree,
        "full_expressions": full_expressions
    }


def optimize_expression(client, full_expressions, gpt_model, x_data, y_data, custom_system_prompt=None, original_f = None, prune_small_terms =True ):
    """
    Optimize and simplify the final expressions.
    
    For each output in the full expression, the function:
      - Prunes and simplifies the expression (replacing floats with parameters and rounding coefficients).
      - Refits the parameters via curve fitting.
      - Plots comparisons with the raw and refitted expressions.
      - Uses an LLM call to further simplify and refit the expression.
    
    Args:
        client: OpenAI client or a compatible client.
        full_expressions: List of full expressions, or just a single expression.
        gpt_model: Model to use for LLM simplification.
        x_data: x data points (numpy array).
        y_data: y data points (numpy array).
        custom_system_prompt: (Optional) Custom system prompt for LLM.
        original_f: (Optional) Original function.
        prune_small_terms: (Optional) Whether to prune small terms. If set to True, the threshold is 1e-6. If set to a float, the threshold is that float.
        
    Returns:
        Dictionary with key 'final_expressions' containing the final refined expressions.
    """
    
    # if custom_system_prompt is None:
    #     custom_system_prompt = """
    #     You are a physics assistant. Your task is to simplify mathematical 
    #     expressions as much as possible while preserving their meaning.
    #     """
    # Handle case where full_expressions is a single expression
    if isinstance(full_expressions, dict) and "full_expressions" in full_expressions:
        full_expressions = full_expressions["full_expressions"]
    elif not isinstance(full_expressions, list):
        full_expressions = [full_expressions]
    Ninputs = x_data.shape[-1] if len(x_data.shape) > 1 else 1

    ranges = (float(np.min(x_data)), float(np.max(x_data)))
    fig, ax = plt.subplots()
    results_all_dicts = []
    
    for i, expr in enumerate(full_expressions):
        final_KAN_expressions = []
        chi_squared_KAN_finals = []
        final_LLM_expressions = []
        chi_squared_LLM_finals = []
        best_chi_squared = float('inf')
        best_expression = None
        best_expression_index = None
        best_fit_type = None  # Track the type of the best fit
        print("###################################################")
        print(f"Simplifying output {i}")
        print("KAN expression (raw):\n", expr)
        f_fitted =eval("lambda x0: "+convert_sympy_to_numpy(expr), {"np": np})
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
        # Calculate chi-squared for the raw expression
        try:
            raw_chi_squared = get_chi_squared(x_data, y_data, f_fitted, [])
            print(f"Raw expression chi-squared: {raw_chi_squared:.4e}")
            # Initialize best values with the raw expression
            best_chi_squared = raw_chi_squared
            best_expression = expr
            best_expression_index = i
            best_fit_type = "raw"  # Initial best fit is the raw expression
        except Exception as e:
            print(f"Error calculating raw chi-squared: {e}")
            best_chi_squared = float('inf')
            best_expression = None
            best_expression_index = None
            best_fit_type = None
    
        # Prune and simplify: replace floats with parameters (round coefficients to 4 digits) then simplify.
        expr = simplify_expression(subst_params(*replace_floats_with_params(expr)),
                                   Ninputs)
        print("KAN expression (simplified):\n", expr)
    
        # Refit parameters.
        expr_np = convert_sympy_to_numpy(expr)
        curve_ansatz_str_np, params_initial = replace_floats_with_params(expr_np)
        print("Converted to numpy, and replaced the new floats with 'params': ", curve_ansatz_str_np)
        curve_ansatz_np = "lambda x0, *params: " + curve_ansatz_str_np
        curve_np = eval(curve_ansatz_np, {"np": np})
        try:
            try:
                params_opt, chi_squared = fit_curve_with_guess(x_data, y_data,
                                                    curve_np, params_initial, try_all_methods=True, log_everything=True)
            except RuntimeError as e:
                #print(f"Refitting failed: {e}. Trying with random initial parameters...")
                # Generate random initial parameters within a reasonable range
                random_params = np.random.uniform(-1.0, 1.0, len(params_initial))
                params_opt, chi_squared = fit_curve_with_guess(x_data, y_data,
                                                    curve_np, random_params, try_all_methods=True, log_everything=True)
            print(f"Refitting: {curve_ansatz_str_np} - so after simplification and refitting gave a chi^2 of {chi_squared:.4e}")
            #simplified_expressions.append(expr)
            #chi_squared_simplified.append(chi_squared)
            
            # Track best chi-squared
            if chi_squared < best_chi_squared:
                best_chi_squared = chi_squared
                best_expression = simplify_expression(subst_params(curve_ansatz_str_np, params_opt), Ninputs) 
                best_expression_index = i
                best_fit_type = "KANsimplified"  # Best fit is from KAN simplification
        except RuntimeError as e:
            params_opt = params_initial
            chi_squared = get_chi_squared(x_data, y_data, curve_np, params_opt)
            print(f"All fits failed, proceeding with unoptimized parameters {e}, chi-squared with unoptimized parameters: {chi_squared:.4e}")
        # Prune and simplify the refitted model.
        if prune_small_terms:
            prune_amount = 1e-6 if prune_small_terms==True else prune_amount
            print(f"Pruning small terms, smaller than {prune_amount}")
            params_opt = [p if abs(p) > prune_amount else 0 for p in params_opt]

        expr_sp = simplify_expression(subst_params(curve_ansatz_str_np, params_opt), Ninputs)
        print("KAN expression (final):\n", expr_sp)
        final_KAN_expressions.append(expr_sp)
        chi_squared_KAN_finals.append(chi_squared)
    
        # Plot comparison.
        f_fitted = eval("lambda x0: "+convert_sympy_to_numpy(expr_sp), {"np": np})
        ax.plot(xs, [f_fitted(x) for x in xs], label="KANSR (refitted)")
    
        # Ask LLM to further simplify the result and refit.
        try:
            expr_llm = call_model_simplify(client, ranges, expr_sp, gpt_model, system_prompt=custom_system_prompt, sympy=True, numpy=False)
            print(f"LLM improvement response is: {expr_llm}")
            curve_ansatz_str_np, params_initial = replace_floats_with_params(convert_sympy_to_numpy(expr_llm))
            curve_ansatz_np = "lambda x0, *params: " + curve_ansatz_str_np
            curve_np = eval(curve_ansatz_np, {"np": np})
            
            try:
                params_opt, chi_squared = fit_curve_with_guess(x_data, y_data,
                                                    curve_np, params_initial, try_all_methods=True, log_everything=True)
            except RuntimeError as e:
                print(f"Refitting failed: {e}. Trying with random initial parameters...")
                # Generate random initial parameters within a reasonable range
                random_params = np.random.uniform(-1.0, 1.0, len(params_initial))
                params_opt, chi_squared = fit_curve_with_guess(x_data, y_data,
                                                    curve_np, random_params, try_all_methods=True, log_everything=True)
            final_LLM_expressions.append(expr_llm)
            chi_squared_LLM_finals.append(chi_squared)
            print(f"LLM improvement gave a chi^2 of {chi_squared:.4e}")
            if prune_small_terms:
                print(f"Pruning small terms, smaller than {prune_amount}")
                params_opt = [p if abs(p) > prune_amount else 0 for p in params_opt]
                expr_final_sp = simplify_expression(subst_params(curve_ansatz_str_np, params_opt), Ninputs)
            else:
                expr_final_sp = simplify_expression(subst_params(curve_ansatz_str_np, params_opt), Ninputs)
            chi_squared = get_chi_squared(x_data, y_data, curve_np, params_opt)
            if chi_squared < best_chi_squared:
                best_chi_squared = chi_squared
                best_expression = expr_final_sp
                best_expression_index = i
                best_fit_type = "LLMsimplified"  # Best fit is from LLM simplification

            # Track best chi-squared
            f_fitted = eval("lambda x0: "+convert_sympy_to_numpy(expr_final_sp), {"np": np})
            ax.plot(xs, [f_fitted(x) for x in xs], label="KANSR (final)")
            print('Final LLM response, simplified and refitted: ', expr_final_sp)
        except Exception as e:
            print(f"Skipping LLM improvement. {e}")
            final_LLM_expressions.append(None)
            chi_squared_LLM_finals.append(None)
    
        ax.legend()
        plt.show()
        print(f"###############################\n# Final formula for output {i}: #\n###############################\n {best_expression} with a chi^2 of {best_chi_squared:.3e} and from the {best_fit_type} fit (of raw/KANsimplified/LLMsimplified)")
    
        result_dict = {
            'raw_expression': expr,
            'final_KAN_expression': final_KAN_expressions,
            'chi_squared_KAN_final': chi_squared_KAN_finals,
            'final_LLM_expression': final_LLM_expressions,
            'chi_squared_LLM_final': chi_squared_LLM_finals,
            'best_expression': best_expression,
            'best_chi_squared': best_chi_squared,
            'best_expression_index': best_expression_index,
            'best_fit_type': best_fit_type  # Add the type of the best fit
        }
        results_all_dicts.append(result_dict)

    best_expressions = [result_dict['best_expression'] for result_dict in results_all_dicts]
    best_chi_squareds = [result_dict['best_chi_squared'] for result_dict in results_all_dicts]
    
    return best_expressions, best_chi_squareds, results_all_dicts

def plot_results(f, ranges, result_dict, model = None, pruned_model = None, title="KAN Symbolic Regression Results"):
    """
    Plot the original function and the approximations.
    
    Args:
        f: Original function
        ranges: Tuple of (min_x, max_x) for the input range
        result_dict: Dictionary with results from optimize_expression
        model: KAN model
        pruned_model: Pruned KAN model
        title: Plot title
        
    Returns:
        matplotlib figure and axes objects
    """
    x = np.linspace(ranges[0], ranges[1], 1000)
    y_true = f(torch.tensor(x).reshape(-1, 1).float()).numpy().flatten()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the true function
    ax.plot(x, y_true, 'k-', label='True Function', linewidth=2)
    
    # Get the best expression
    best_expr = result_dict['best_expression']
    best_chi_squared = result_dict['best_chi_squared']
    
    # Try to plot the best expression
    try:
        y_best = eval(convert_sympy_to_numpy(best_expr), {"np": np, "x0": x})
        ax.plot(x, y_best, 'r--', label=f'Best Expression (χ²={best_chi_squared:.5e})', linewidth=2)
    except Exception as e:
        print(f"Error plotting best expression: {e}")
    
    # Try to plot the simplified expression
    try:
        best_idx = result_dict['best_expression_index']
        raw_expr = result_dict['raw_expression']
        y_raw = eval(convert_sympy_to_numpy(raw_expr), {"np": np, "x0": x})
        ax.plot(x, y_raw, 'k--', label=f'Raw expression from LLMSR', linewidth=2)


        simplified_by_KAN_expr = result_dict['final_KAN_expression'][best_idx]
        chi_squared = result_dict['chi_squared_KAN_final'][best_idx]
        y_simplified = eval(convert_sympy_to_numpy(simplified_by_KAN_expr), {"np": np, "x0": x})
        ax.plot(x, y_simplified, 'g-.', label=f'Simplified by KAN (χ²={chi_squared:.5e})', linewidth=2)

        simplified_by_LLM_expr = result_dict['final_LLM_expression'][best_idx]
        chi_squared = result_dict['chi_squared_LLM_final'][best_idx]
        
        y_simplified = eval(convert_sympy_to_numpy(simplified_by_LLM_expr), {"np": np, "x0": x})
        ax.plot(x, y_simplified, 'g-.', label=f'Simplified by LLM and refitted (χ²={chi_squared:.5e})', linewidth=2)
    except Exception as e:
        print(f"Error plotting simplified expression: {e}")
    # Try to plot the model and pruned model predictions
    if model is not None:
        model_preds = model(torch.tensor(x).reshape(-1, 1).float()).detach().numpy().flatten()
        ax.plot(x, model_preds, 'b:', label='KAN Model', linewidth=2)
    if pruned_model is not None:
        pruned_preds = pruned_model(torch.tensor(x).reshape(-1, 1).float()).detach().numpy().flatten()
        ax.plot(x, pruned_preds, 'm:', label='Pruned KAN Model', linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def run_complete_pipeline(client, f, ranges=(-np.pi, np.pi), width=[1,4,1], grid=7, k=3, 
                         train_steps=50, generations=3, gpt_model="openai/gpt-4o", device='cpu',
                         node_th=0.2, edge_th=0.2, custom_system_prompt_for_second_simplification=None, optimizer="LBFGS", population=10, temperature=0.1,
                        exit_condition=None, verbose=0, use_async=True, plot_fit=True, plot_parents=False, demonstrate_parent_plotting=False):
    """
    Run the complete KAN symbolic regression pipeline on a univariate function.
    
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
        - 'result_dict': Result dictionary
    """
    # Validate input architecture
    if width[0] != 1:
        raise ValueError(f"First layer width must be 1 for univariate function, got {width[0]}")
    # 1. Create the model
    try:
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
                                    exit_condition=exit_condition if exit_condition is not None else min(train_loss).item(), verbose=verbose, use_async=use_async, plot_fit=plot_fit, plot_parents=plot_parents, demonstrate_parent_plotting=demonstrate_parent_plotting)
        symb_expr_sorted = sort_symb_expr(res)
        
        # 5. Build expression tree
        node_data = build_expression_tree(pruned_model, symb_expr_sorted, top_k=3)
        # 6. Optimize expression
        # Convert training data to numpy arrays for optimization
        x_data = dataset['train_input'].cpu().numpy().flatten()
        y_data = dataset['train_label'].cpu().numpy().flatten()
        # Optimize and simplify the expression
        best_expressions, best_chi_squareds, result_dicts = optimize_expression(
            client, node_data, gpt_model, x_data, y_data, 
            custom_system_prompt=custom_system_prompt_for_second_simplification, original_f=f, prune_small_terms=True
        )
        result_dict = result_dicts[0]

        # Print the results
        best_index = result_dict['best_expression_index']
        print(f"best expression: {result_dict['best_expression']}, at index {best_index}, with chi^2 {result_dict['best_chi_squared']}")
        print(f"initially: {result_dict['raw_expression'][best_index]}")
        print(f"refitting all coefficients in KAN: {result_dict['final_KAN_expression'][best_index]}, chi^2 {result_dict['chi_squared_KAN_final'][best_index]}")
        print(f"simplifying by LLM and refitting again: {result_dict['final_LLM_expression'][best_index]}, chi^2 {result_dict['chi_squared_LLM_final'][best_index]}")
        return {
            'trained_model': model,
            'pruned_model': pruned_model,
            'train_loss': train_loss,
            'symbolic_expressions': symb_expr_sorted,
            'node_tree': node_data,
            'result_dict': result_dict,
            'dataset': dataset,
            'best_expressions': best_expressions,
            'best_chi_squareds': best_chi_squareds
        }
    except Exception as e:
        # Return partial results based on what was completed
        results = {}
        if 'model' in locals():
            results['trained_model'] = model
        if 'pruned_model' in locals():
            results['pruned_model'] = pruned_model
        if 'train_loss' in locals():
            results['train_loss'] = train_loss
        if 'symb_expr_sorted' in locals():
            results['symbolic_expressions'] = symb_expr_sorted
        if 'node_data' in locals():
            results['node_tree'] = node_data
        if 'result_dict' in locals():
            results['result_dict'] = result_dict
        if 'dataset' in locals():
            results['dataset'] = dataset
        if 'best_expressions' in locals():
            results['best_expressions'] = best_expressions
            results['best_chi_squareds'] = best_chi_squareds
        print(f"Error in pipeline: {e}, returning partial results: {list(results.keys())}")
        return results
