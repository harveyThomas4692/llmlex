"""
DEPRECATED: This module has been replaced by the class-based implementation in kansr.py.

This file is kept for backwards compatibility but should not be used for new code.
All functionality has been moved to the KANSR class in kansr.py.
"""

import warnings

warnings.warn(
    "The kan_sr module is deprecated. Please use the KANSR class from kansr module instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new implementation for backward compatibility
from LLMSR.kansr import KANSR, run_complete_pipeline

# Re-export all functions from the old module through the new implementation
# This allows existing code to continue working while encouraging migration to the new API

def create_kan_model(width, grid, k, seed=17, symbolic_enabled=False, device='cpu'):
    """DEPRECATED: Use KANSR class instead."""
    warnings.warn(
        "create_kan_model is deprecated. Use KANSR class constructor instead.",
        DeprecationWarning,
        stacklevel=2
    )
    kan = KANSR(width=width, grid=grid, k=k, seed=seed, 
                symbolic_enabled=symbolic_enabled, device=device)
    return kan.raw_model

def subst_params(a, p):
    """DEPRECATED: Use KANSR._subst_params instead."""
    warnings.warn(
        "subst_params is deprecated. Use KANSR class methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Create a temporary KANSR instance to access the method
    tmp = KANSR(model=None, width=[1,1,1], grid=1, k=1)
    return tmp._subst_params(a, p)

def convert_sympy_to_numpy(expr):
    """DEPRECATED: Use KANSR._convert_sympy_to_numpy instead."""
    warnings.warn(
        "convert_sympy_to_numpy is deprecated. Use KANSR class methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Create a temporary KANSR instance to access the method
    tmp = KANSR(model=None, width=[1,1,1], grid=1, k=1)
    return tmp._convert_sympy_to_numpy(expr)

def simplify_expression(formula, N=10, timeout=30):
    """DEPRECATED: Use KANSR._simplify_expression instead."""
    warnings.warn(
        "simplify_expression is deprecated. Use KANSR class methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Create a temporary KANSR instance to access the method
    tmp = KANSR(model=None, width=[1,1,1], grid=1, k=1)
    return tmp._simplify_expression(formula, N, timeout)

def replace_floats_with_params(expr_str):
    """DEPRECATED: Use KANSR._replace_floats_with_params instead."""
    warnings.warn(
        "replace_floats_with_params is deprecated. Use KANSR class methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Create a temporary KANSR instance to access the method
    tmp = KANSR(model=None, width=[1,1,1], grid=1, k=1)
    return tmp._replace_floats_with_params(expr_str)

def call_model_simplify(client, ranges, expr, gpt_model="openai/gpt-4o", system_prompt=None, sympy=True, numpy=False, num_answers=3):
    """DEPRECATED: Use KANSR._call_model_simplify instead."""
    warnings.warn(
        "call_model_simplify is deprecated. Use KANSR class methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Create a temporary KANSR instance to access the method
    tmp = KANSR(model=None, width=[1,1,1], grid=1, k=1)
    return tmp._call_model_simplify(client, ranges, expr, gpt_model, system_prompt, sympy, numpy, num_answers)

def sort_symb_expr(symb_expr):
    """DEPRECATED: Use KANSR._sort_symbolic_expressions instead."""
    warnings.warn(
        "sort_symb_expr is deprecated. Use KANSR class methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Create a temporary KANSR instance to access the method
    tmp = KANSR(model=None, width=[1,1,1], grid=1, k=1)
    return tmp._sort_symbolic_expressions(symb_expr)

def build_expression_tree(model, symb_expr_sorted, top_k=3):
    """DEPRECATED: Use KANSR.build_expression_tree instead."""
    warnings.warn(
        "build_expression_tree is deprecated. Use KANSR.build_expression_tree instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Create a temporary KANSR instance to access the method
    tmp = KANSR(model=model, width=None, grid=None, k=None)
    tmp.symbolic_expressions = symb_expr_sorted
    return tmp.build_expression_tree(top_k)

def optimize_expression(client, full_expressions, gpt_model, x_data, y_data, custom_system_prompt=None, original_f=None, prune_small_terms=True, plot_all=True, num_prompts_per_attempt=3, timeout_simplify=10):
    """DEPRECATED: Use KANSR.optimize_expressions instead."""
    warnings.warn(
        "optimize_expression is deprecated. Use KANSR.optimize_expressions instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Create a KANSR instance with the minimal parameters
    tmp = KANSR(model=None, width=[1,1,1], grid=1, k=1)
    
    # Set up the node_tree if full_expressions is a dict
    if isinstance(full_expressions, dict) and "full_expressions" in full_expressions:
        tmp.node_tree = full_expressions
        full_expressions = full_expressions["full_expressions"]
    else:
        # Create a minimal node_tree with just the full_expressions
        tmp.node_tree = {"full_expressions": full_expressions if isinstance(full_expressions, list) else [full_expressions]}
    
    return tmp.optimize_expressions(
        client, gpt_model, x_data, y_data, 
        custom_system_prompt=custom_system_prompt, 
        original_f=original_f, 
        prune_small_terms=prune_small_terms, 
        plot_all=plot_all,
        num_prompts_per_attempt=num_prompts_per_attempt, 
        timeout_simplify=timeout_simplify
    )

def plot_results(f, ranges, result_dict, model=None, pruned_model=None, title="KAN Symbolic Regression Results", plotmaxmin=[[None, None], [None, None]]):
    """DEPRECATED: Use KANSR.plot_results instead."""
    warnings.warn(
        "plot_results is deprecated. Use KANSR.plot_results instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Create a temporary KANSR instance
    tmp = KANSR(model=model, width=None, grid=None, k=None)
    if model is not None:
        tmp.raw_model = model
    if pruned_model is not None:
        tmp.model = pruned_model
    return tmp.plot_results(f, ranges, result_dict, title, plotmaxmin)