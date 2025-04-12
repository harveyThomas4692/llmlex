"""
Module for symbolic regression using Kolmogorov-Arnold Networks (KANs).

This module provides a class-based interface for training KAN models on data,
extracting symbolic expressions from the trained models,
simplifying these expressions, and fitting them to data.
"""

import numpy as np
import torch
import sympy as sp
from sympy import symbols, simplify, sin, cos, exp, log, sqrt, sinh, cosh, tanh
from sympy.printing.numpy import NumPyPrinter
import matplotlib.pyplot as plt
import re
import copy
import asyncio
import concurrent.futures
import io
import stopit
import logging
from kan import KAN, create_dataset

import llmlex.llmlex as LLMLEx
from llmlex.fit import get_n_chi_squared, fit_curve_with_guess, fit_curve_with_guess_jax, test_np_function_equivalence, get_n_chi_squared_from_predictions, test_data_equivalence
import llmlex.llm as llm
import tqdm

class KANLEX:
    """
    A class for performing symbolic regression using Kolmogorov-Arnold Networks (KANs).
    
    This class provides methods for training KAN models, converting them to symbolic expressions,
    simplifying and optimising these expressions, and visualizing results.
    """
    
    def __init__(self, client = None, width=None, grid=None, k=None, seed=17, symbolic_enabled=False, 
                 device='cpu', log_level=logging.INFO, model=None):
        """
        Initialize a KAN_LEx instance.
        
        Args:
            client: Client for LLM API calls
            width: List specifying the network architecture (e.g., [1,4,1])
            grid: Grid size for KAN
            k: Number of basis functions
            seed: Random seed for reproducibility
            symbolic_enabled: Whether to enable symbolic features
            device: Device to use ('cpu' or 'cuda')
            log_level: Logging level
            model: Pre-existing KAN model (optional, if provided width/grid/k are ignored)
        """
        # Set up logging
        self.logger = logging.getLogger("LLMLEx.kanLEx")
        self.logger.setLevel(log_level)
        self.logger.propagate = False  # Prevent propagation to parent loggers
        
        # Only add a handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        # Initialize model
        if model is not None:
            self.raw_model = model
        elif width is not None and grid is not None and k is not None:
            self.raw_model = self._create_kan_model(width, grid, k, seed, symbolic_enabled, device)
        else:
            raise ValueError("Either model or (width, grid, k) must be provided")

        if client is None:
            raise ValueError("Client must be provided")
            
        self.device = self.raw_model.device
        self.model = None  # Will hold pruned model if prune is True
        self.dataset = None
        self.training_history = None
        self.final_train_loss = None
        self.symbolic_expressions = None
        self.expression_tree = None
        self.client = client
        self.f = None
        
        
        # Function mapping dictionaries
        self.numpy_to_sympy = {
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

    def _create_kan_model(self, width, grid, k, seed=17, symbolic_enabled=False, device='cpu'):
        """
        Create a KAN model with specified parameters.
        
        Args:
            width: List specifying the network architecture
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
        
        model = KAN(width=width, grid=grid, k=k, seed=seed, device=device, symbolic_enabled=symbolic_enabled)
        return model
    
    def create_dataset(self, f, ranges=(-np.pi, np.pi), n_var=1, train_num=10000, test_num=1000):
        """
        Create a dataset for training the KAN model. Assigns an original function f to the instance.
        
        Args:
            f: Target function to approximate
            ranges: Tuple of (min, max) for the input range
            n_var: Number of input variables
            train_num: Number of training samples
            test_num: Number of test samples
            
        Returns:
            Dataset dictionary containing training and test data
        """
        self.dataset = create_dataset(f, n_var=n_var, ranges=ranges,
                                     train_num=train_num, test_num=test_num,
                                     device=self.device)
        self.f = f
        return self.dataset
    
    def train_kan(self, dataset=None, opt="LBFGS", steps=50, prune=True, node_th=0.2, edge_th=0.2, **kwargs):
        """
        Train the KAN model on the provided dataset.
        
        Args:
            dataset: Dataset to train on (if None, uses self.dataset)
            opt: Optimiser to use
            steps: Number of training steps
            prune: Whether to prune the model after training
            node_th: Node threshold for pruning
            edge_th: Edge threshold for pruning
            
        Returns:
            Dictionary containing training results
        """
        if dataset is not None:
            self.dataset = dataset
        
        if self.dataset is None:
            raise ValueError("No dataset provided. Call create_dataset first or provide a dataset.")
        
        self.logger.info(f"Training KAN model with {opt} optimiser for {steps} steps")
        self.training_history = self.raw_model.fit(self.dataset, opt=opt, steps=steps, **kwargs)
        print(f"Unpruned model. Pruning? {prune}")
        self.raw_model.plot()
        if prune:
            self.logger.info(f"Pruning model with node_th={node_th}, edge_th={edge_th}")
            self.model = self.raw_model.prune(node_th=node_th, edge_th=edge_th)
            self.logger.info("Pruned model:")
            self.model.plot()
        else:
            self.model = self.raw_model
        
        self.final_train_loss = self.training_history['train_loss'][-1].item()
        self.logger.info(f"Final train loss: {self.final_train_loss}")
        return self.final_train_loss
    
    def get_symbolic(self, client=None, population=10, generations=3, temperature=0.1, 
                           gpt_model="openai/gpt-4o", exit_condition=None, verbose=0, 
                           use_async=True, plot_fit=True, plot_parents=False, demonstrate_parent_plotting=False, constant_on_failure=False,
                           num_answers_per_prompt=3, timeout_simplify=10, custom_system_prompt_for_second_simplification=None,
                           prune_small_terms=True, plot_all=True, original_f=None, number_of_prompts=3, simplification_gpt_model=None, imports=None):
        """
        Convert the trained KAN model to symbolic expressions.
        
        Args:
            client: OpenAI client or compatible client
            population: Population size for genetic algorithm
            generations: Number of generations for the genetic algorithm
            temperature: Temperature parameter for the genetic algorithm
            gpt_model: GPT model to use for simplification
            exit_condition: Lowest absolute value of score for the genetic algorithm to stop
            verbose: Verbosity level
            use_async: Whether to use async execution
            plot_fit: Whether to plot fitting results
            plot_parents: Whether to plot parent solutions
            number_of_prompts: Number of prompts to try in the simplification step
            num_answers_per_prompt: Number of answers per prompt in the simplification step
            timeout_simplify: Timeout for simplification in seconds
            custom_system_prompt_for_second_simplification: Custom system prompt for the second simplification step
            prune_small_terms: Whether to prune small terms
            plot_all: Whether to plot all results after optimisation
            original_f: Original function for comparison (optional)
            demonstrate_parent_plotting: Whether to demonstrate parent plotting
            constant_on_failure: Whether to use a constant on failure
            simplification_gpt_model: GPT model to use for simplification
            imports: List of import statements to include in the prompt (optional)

        Returns:
            List of best expressions
            list of best n_chi-squared values
            list of result dictionaries containing detailed information about all expressions and their optimisations
            list of all results, with each entry sorted by n_chi-squared
        """
        initial_usage = llm.check_key_usage(self.client)
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_kan() first.")
        if original_f is None:
            original_f = self.f
        if simplification_gpt_model is None:
            simplification_gpt_model = gpt_model
            
        # Use provided client or instance client
        client_to_use = client if client is not None else self.client
        if client_to_use is None:
            raise ValueError("Client must be provided either during initialization or to this method call")
            
        # Update instance client
        self.client = client_to_use
            
        # Use n_chi_squared from model predictions as exit condition if not specified
        if exit_condition is None:
            # try:
            #     x_data = self.dataset['train_input'].cpu().numpy()
            #     y_data = self.dataset['train_label'].cpu().numpy()
            #     predictions = self.model(torch.tensor(x_data).float()).detach().cpu().numpy()
            #     exit_condition = get_n_chi_squared_from_predictions(x_data, y_data, predictions)
            #     self.logger.info(f"Using n_chi_squared from entire model predictions as exit condition: {exit_condition}")
            # except Exception as e:
            #     self.logger.warning(f"Could not calculate n_chi_squared: {e}. Using default exit condition.")
            #     exit_condition = 1e-3
            exit_condition = 1e-3
        # Log warning about using default exit condition
        self.logger.warning(f"Using default exit condition of {exit_condition}. Consider passing 'overall_nchi_squared'"
            f"as an exit condition - this is the n_chi_squared of the entire model on its inputs. "
            f"It's not the default as this is not necessarily meaningful for each individual activation function.")

        self.logger.info(f"Converting KAN model to symbolic expressions (exit_condition={exit_condition})")
        result_of_kan_to_symbolic = LLMLEx.kan_to_symbolic(
            self.model, client_to_use, population=population, generations=generations,
            temperature=temperature, gpt_model=gpt_model, exit_condition=exit_condition,
            verbose=verbose, use_async=use_async, plot_fit=plot_fit, plot_parents=plot_parents,
            demonstrate_parent_plotting=demonstrate_parent_plotting, constant_on_failure=constant_on_failure,
            imports=imports
        )
        
        self.symbolic_expressions = self._sort_symbolic_expressions(result_of_kan_to_symbolic)
        self.build_expression_tree()
        self.optimised_expressions = self.optimise_expressions(client_to_use, simplification_gpt_model, x_data=self.dataset['train_input'].cpu().numpy(), y_data=self.dataset['train_label'].cpu().numpy(),
                                                                custom_system_prompt=custom_system_prompt_for_second_simplification,
                                                                prune_small_terms=prune_small_terms, plot_all=plot_all, original_f=original_f,
                                                                num_answers_per_prompt=num_answers_per_prompt, timeout_simplify=timeout_simplify, number_of_prompts=number_of_prompts)
        best_expressions, best_n_chi_squareds, results_all_dicts, all_results_sorted = self.optimised_expressions
        self.results_all_dicts = results_all_dicts
        
        final_usage = llm.check_key_usage(self.client)
        cost = f"${(final_usage - initial_usage):.2f}" if isinstance(final_usage, (float, int)) and isinstance(initial_usage, (float, int)) else 'unknown'
        self.logger.info(f"API key usage whilst this get_symbolic was running: {cost}")

        return best_expressions, best_n_chi_squareds, results_all_dicts, all_results_sorted
    
    def _sort_symbolic_expressions(self, symb_expr):
        """
        Sort symbolic expressions by score.
        
        Args:
            symb_expr: Dictionary of symbolic expressions
        
        Returns:
            Sorted dictionary of symbolic expressions
        """
        symb_expr_sorted = {}
        # Build dictionary with all expressions ordered by score
        for kan_conn, sub_res in symb_expr.items():
            if sub_res is None:
                self.logger.warning(f"Could not fit a function for connection {kan_conn}")
                continue
            ordered_elements = sorted([item for sublist in sub_res for item in sublist], key=lambda item: -item['score'])
            symb_expr_sorted[kan_conn] = ordered_elements
            self.logger.info(f"Approximation for {kan_conn}: {ordered_elements[0]['ansatz'].strip()}, has parameters {np.round(ordered_elements[0]['params'], 1)}")
        self.symbolic_expressions = symb_expr_sorted
        return symb_expr_sorted

    def build_expression_tree(self, top_k=3):
        """
        Build an expression tree from KAN model connections.
        
        Args:
            top_k: Number of candidate expressions to retain per connection
            
        Returns:
            Dictionary containing edge expressions, node tree, and full expressions
        """
        if self.symbolic_expressions is None:
            raise ValueError("Symbolic expressions not generated yet. Call convert_to_symbolic() first.")
            
        self.logger.info("Building expression tree")
        # Process each connection to select the best candidate expression
        edge_dict = {}
        top_k_edge_dicts = {}
        
        for kan_conn, sub_res in self.symbolic_expressions.items():
            best_expr = None
            best_score = -np.inf
            top_k_candidates = []
            
            if sub_res is None or not isinstance(sub_res, list):
                self.logger.warning(f"Could not fit a function for connection {kan_conn}")
                continue
            
            # Handle case where list is empty
            if len(sub_res) == 0:
                self.logger.warning(f"Empty result list for connection {kan_conn}")
                continue
                
            # Limit to top_k or maximum available
            candidates_to_process = sub_res[:min(top_k, len(sub_res))]
            for candidate in candidates_to_process:
                # Clean up the ansatz string
                ansatz = candidate['ansatz'].replace('*x', ' * x') \
                                          .replace('(x)', '(1. * x)') \
                                          .replace('-x', '-1. * x').replace('x,',' x ,') \
                                          .replace('(x', '( x').replace('x)', 'x )') \
                                          .replace('/x', '/ x').strip()
                if "lambda" in ansatz:
                    continue  # Skip lambda functions
                    
                score = candidate['score']
                expr = self._subst_params(ansatz, candidate['params'], round_to=16)
                
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
                self.logger.info(f"KAN Connection: {kan_conn}, Best Expression: {best_expr}, Score: {best_score:.5f}")
            else:
                self.logger.warning(f"Could not fit a function for connection {kan_conn}")
        
        # Build node tree: each node (l, n) is the sum over incoming edges
        node_tree = {}
        for l in range(len(self.model.width_in) - 1):
            for n in range(self.model.width_in[l+1]):
                node_tree[(l, n)] = " + ".join([
                    edge_dict.get((l, c, n), "").replace(' x', f' x[{l-1},{c}]' if l > 0 else f' x{c}')
                    for c in range(self.model.width_out[l])
                    if edge_dict.get((l, c, n), "") != ""
                ])
        
        # Clean up the expressions
        for k, v in node_tree.items():
            node_tree[k] = v.replace('+ -', '- ')
        
        # Build full expression, prepopulate with output nodes
        full_expressions = []
        for o in range(self.model.width_in[-1]):
            res = node_tree[(len(self.model.width_in) - 2, o)]
            # Traverse down lower layers
            for l in list(range(len(self.model.width_in) - 2))[::-1]:
                n_count = len([x for x in node_tree.keys() if x[0] == l])
                for n in range(n_count):
                    res = res.replace(f'x[{l},{n}]', f'({node_tree[(l, n)]})')
            full_expressions.append(self._simplify_expression(res, self.model.width_in[0] - 1))
        
        self.expression_tree = {
            "edge_dict": edge_dict,
            "top_k_edge_dicts": top_k_edge_dicts,
            "node_tree": node_tree,
            "full_expressions": full_expressions
        }
        
        return self.expression_tree
    
    def _validate_optimisation_prerequisites(self, x_data, y_data):
        """Validate that prerequisites for optimisation are met."""
        if self.expression_tree is None:
            raise ValueError("Expression tree not built yet. Call build_expression_tree() first.")
        
        if x_data is None or y_data is None:
            if self.dataset is None:
                raise ValueError("No dataset available. Provide x_data and y_data or train the model first.")
            
    def _prepare_optimisation_data(self, x_data, y_data):
        """Prepare data for optimisation."""
        if x_data is None or y_data is None:
            x_data = self.dataset['train_input'].cpu().numpy()
            y_data = self.dataset['train_label'].cpu().numpy()
        
        full_expressions = self.expression_tree["full_expressions"]
        
        # Handle case where full_expressions is a single expression
        if not isinstance(full_expressions, list):
            full_expressions = [full_expressions]
        
        Ninputs = x_data.shape[-1] if len(x_data.shape) > 1 else 1
        
        # Create ranges for each input dimension
        if len(x_data.shape) > 1:
            ranges = [(float(np.min(x_data[:, i])), float(np.max(x_data[:, i]))) for i in range(Ninputs)]
        else:
            ranges = [(float(np.min(x_data)), float(np.max(x_data)))]
        
        return x_data, y_data, full_expressions, Ninputs, ranges

    def _setup_plotting(self, plot_all, ranges, Ninputs, original_f=None, xs=None):
        """Setup plotting if required."""
        if not plot_all:
            return None, None, None
        
        fig, ax = plt.subplots()
        
        # Create xs for plotting if not provided
        if xs is None:
            if Ninputs == 1:
                xs = np.linspace(ranges[0][0], ranges[0][1], 100)
            else:
                xs = np.arange(ranges[0][0], ranges[0][1], (ranges[0][1]-ranges[0][0])/100)
        
        # Plot original function if provided
        if original_f is not None:
            try:
                try:
                    ax.plot(xs, [original_f(torch.tensor(x)) for x in xs], label="function we're fitting", linewidth=4, alpha=0.5, color="black")
                except TypeError:
                    ax.plot(xs, [original_f(x) for x in xs], label="function we're fitting", linewidth=4, alpha=0.5, color="black")
            except Exception as e:
                self.logger.warning(f"Original function 'f' not defined; skipping plotting actual function: {e}")
        
        return fig, ax, xs

    def _process_raw_expression(self, expr, x_data, y_data, Ninputs, ranges):
        """Process and evaluate the raw expression."""
        
        # Test raw expression
        expr_raw_float_np = self._convert_sympy_to_numpy(expr)
        f_fitted = self._convert_to_np_function(expr_raw_float_np, Ninputs, floats_only=True)
        
        # Calculate n_chi-squared for the raw expression
        try:
            raw_n_chi_squared = get_n_chi_squared(x_data, y_data, f_fitted, [], explain_if_inf=True, string_for_explanation=expr_raw_float_np)
        except Exception as e:
            self.logger.warning(f"Error calculating raw n_chi-squared: {e}")
            raw_n_chi_squared = float('inf')

        self.logger.info(f"KAN expression (raw), n_chi2 with original data: {raw_n_chi_squared:.4e}: {self._round_floats_in_expression(expr)}")
        
        return {
            'expression': expr,
            'expression_numpy': expr_raw_float_np,
            'n_chi_squared': raw_n_chi_squared,
            'fit_type': 'raw'
        }

    def _plot_expression(self, ax, xs, result, label_prefix):
        """Plot an expression result."""
        if ax is None or result['expression'] is None:
            return
        Ninputs = xs.shape[-1] if len(xs.shape) > 1 else 1
        try:
            f_fitted, _ = self._convert_to_np_function(result['expression_numpy'], Ninputs, params=True, return_string=True)
            ax.plot(xs, [f_fitted(x) for x in xs], label=f"{label_prefix} n_chi2: {result['n_chi_squared']:.4e}")
        except Exception as e:
            self.logger.warning(f"Error plotting {label_prefix} expression: {e}")

    def _update_best_result(self, best_result, new_result):
        """Update the best result if the new result is better."""
        if new_result['expression'] is None or new_result['n_chi_squared'] >= best_result['n_chi_squared']:
            return False
        
        best_result['n_chi_squared'] = new_result['n_chi_squared']
        best_result['expression'] = new_result['expression']
        best_result['expression_numpy'] = new_result['expression_numpy']
        best_result['fit_type'] = new_result['fit_type']
        return True

    def _is_relatively_close(self, a, b, atol=1e-2):
        """
        Check if two values are relatively close or both small.
    
        Args:
            a: First value
            b: Second value
            atol: Absolute tolerance for comparison

        Returns:
            bool: True if values are close or both small
        """
        # Check if both values are small (near zero)
        if abs(a) < atol and abs(b) < atol:
            return True
    
        # Check if they're relatively close
        if abs(a) > atol or abs(b) > atol:
            return abs(a - b) / max(abs(a), abs(b)) < atol
    
        return True

    def _verify_expression_equivalence(self, original_expr, simplified_expr, x_data, y_data, 
                                      description="simplified", atol=1e-2, check_functional_equivalence=True):
        """
        Verify that a simplified expression is functionally equivalent to the original.
        
        Args:
            original_expr: Original expression in numpy-compatible format, should be explicitly float
            simplified_expr: Simplified expression in numpy-compatible format, should be explicitly float
            x_data: Input data for testing
            y_data: Output data for testing
            description: Description of the simplified expression for logging
            atol: Absolute tolerance for comparing chi-squared values
            check_functional_equivalence: Whether to check function output equivalence across input space
            
        Returns:
            tuple: (is_equivalent, detailed_info, chi_squared)
                - is_equivalent: Boolean indicating if expressions are functionally equivalent
                - detailed_info: Additional information about the verification
                - chi_squared: The chi-squared value of the simplified expression
        """
        Ninputs = x_data.shape[-1] if len(x_data.shape) > 1 else 1
        # Evaluate chi-squared of the simplified expression
        try:
            f_simplified = self._convert_to_np_function(simplified_expr, Ninputs, floats_only=True)
            simplified_n_chi_squared = get_n_chi_squared(
                x_data, y_data, f_simplified, [], 
                explain_if_inf=True, string_for_explanation=simplified_expr
            )
            
            f_original = self._convert_to_np_function(original_expr, Ninputs, floats_only=True)
            original_n_chi_squared = get_n_chi_squared(
                x_data, y_data, f_original, [], 
                explain_if_inf=True, string_for_explanation=original_expr
            )
                
            # Check if chi-squared values are close
            chi_squared_equivalent = self._is_relatively_close(simplified_n_chi_squared, original_n_chi_squared, atol)
            
            if not chi_squared_equivalent:
                self.logger.error(f"Problem with {description}, n_chi2 {simplified_n_chi_squared:.4e} is not close to original n_chi2 {original_n_chi_squared:.4e}")
            else:
                self.logger.info(f"Expression ({description}) maintains similar n_chi2: {simplified_n_chi_squared:.4e}")
            
            # Check if the expressions are functionally equivalent (output similar values)
            if check_functional_equivalence:
                original_expr_to_check = original_expr
                if isinstance(original_expr, dict) and 'expression_numpy' in original_expr:
                    original_expr_to_check = original_expr['expression_numpy']
                
                is_equivalent, diff_info = test_np_function_equivalence(
                    self._convert_to_np_function(original_expr_to_check, Ninputs, floats_only=True), 
                    self._convert_to_np_function(simplified_expr, Ninputs, floats_only=True), 
                    x_data
                )
                
                if not is_equivalent:
                    if isinstance(diff_info, (int, float)):
                        self.logger.info(f"{description.capitalize()} expression differs from original by average relative difference: {diff_info:.4e}")
                    else:
                        self.logger.info(f"Could not compare {description} with original: {diff_info}")
                        
                    return False, diff_info, simplified_n_chi_squared
            else:
                is_equivalent = chi_squared_equivalent
                diff_info = abs(simplified_n_chi_squared - original_n_chi_squared)
                
            return is_equivalent, diff_info, simplified_n_chi_squared
            
        except Exception as e:
            self.logger.error(f"Error verifying equivalence of {description} expression: {e}, original: {self._round_floats_in_expression(original_expr)}, simplified: {self._round_floats_in_expression(simplified_expr)}")
            return False, str(e), float('inf')

    def _process_and_refit_raw_expression(self, expr, x_data, y_data, Ninputs,
                                             timeout_simplify, prune_small_terms, raw_result):
        """Process and evaluate raw expression."""
        # Prune and simplify: replace floats with parameters then simplify
        expr_simp_float_sp = self._simplify_expression(
            self._subst_params(*self._replace_floats_with_params(expr), round_to=16),
            Ninputs, timeout=timeout_simplify*3
        )
        
        # Convert simplified expression to numpy format
        expr_simp_float_np = self._convert_sympy_to_numpy(expr_simp_float_sp)
        
        # Verify that simplified expression is equivalent to original
        is_equivalent, diff_info, simplified_expr_n_chi_squared = self._verify_expression_equivalence(
            raw_result['expression_numpy'], expr_simp_float_np, 
            x_data, y_data, 
            description="raw expression (simplified)"
        )
        
        # Refit parameters
        curve_ansatz_str_param, params_initial = self._replace_floats_with_params(expr_simp_float_np)
        curve_np, curve_ansatz_np = self._convert_to_np_function(curve_ansatz_str_param, Ninputs, params=True, return_string=True)
        
        n_chi_squared_after_refitting = float('inf')
        params_opt = params_initial
        
        try:
            self.logger.info(f"Refitting simplified expression, trying to improve n_chi2 from {simplified_expr_n_chi_squared:.4e}, {curve_ansatz_str_param}, {params_initial}, {curve_np}, {curve_ansatz_np}.")
            params_opt, n_chi_squared_after_refitting = self._fit_params(
                x_data, y_data, curve_np, params_initial, curve_ansatz_np, 
                log_methods=True, log_everything=True, try_harder_jax=True
            )
            self.logger.info(f"Refitting, new n_chi2: {n_chi_squared_after_refitting:.4e} for {curve_ansatz_str_param}")
            
            # # Process successful fitting result
            expr_simp_with_fitted_floats_sp = self._simplify_expression(
                self._subst_params(curve_ansatz_str_param, params_opt), 
                Ninputs, timeout=timeout_simplify*3
            )
            expr_simp_with_fitted_floats_np = self._convert_sympy_to_numpy(expr_simp_with_fitted_floats_sp)
            # kan_result = {
            #     'expression': expr_simp_with_fitted_floats_sp,
            #     'expression_numpy': expr_simp_with_fitted_floats_np,
            #     'n_chi_squared': n_chi_squared_after_refitting,
            #     'fit_type': 'KANsimplified'
            # }
        except RuntimeError as e:
            params_opt = params_initial
            n_chi_squared_after_refitting = get_n_chi_squared(
                x_data, y_data, curve_np, params_opt, 
                explain_if_inf=True, string_for_explanation=curve_ansatz_np
            )
            expr_simp_with_fitted_floats_sp = expr_simp_float_sp
            expr_simp_with_fitted_floats_np = expr_simp_float_np
            self.logger.warning(f"All fits failed: {e}, n_chi-squared with unoptimised parameters: {n_chi_squared_after_refitting:.4e}")
        
        # Apply pruning if requested
        if prune_small_terms:
            prune_amount = 1e-6 if prune_small_terms is True else prune_small_terms
            self.logger.info(f"Pruning small terms, smaller than {prune_amount}")
            params_opt = self._prune_small_params(params_opt, prune_amount)
        
        # Final simplified expression
        expr_final_float_sp = self._simplify_expression(
            self._subst_params(curve_ansatz_str_param, params_opt, round_to=16), Ninputs
        )
        expr_final_float_np = self._convert_sympy_to_numpy(expr_final_float_sp)
        
        # Verify final expression is equivalent to the refitted expression
        is_final_equivalent, final_diff_info, n_chi_squared_refitted_final = self._verify_expression_equivalence(
            expr_simp_with_fitted_floats_np,# original
            expr_final_float_np,# simplified
            x_data, y_data,
            description="refitted vs refitted and pruned",
            check_functional_equivalence=False  # Just compare chi-squared values
        )
        
        self.logger.info(f"raw refitted and pruned expression (n_chi2: {n_chi_squared_refitted_final:.4e}): {self._round_floats_in_expression(expr_final_float_sp)}")

        
        # Return final KAN expression result
        return {
            'expression': expr_final_float_sp,
            'expression_numpy': expr_final_float_np,
            'n_chi_squared': n_chi_squared_refitted_final,
            'fit_type': 'rawrefitted'
        }
    
    def _round_floats_in_expression(self, expr):
        """Round all floating point numbers in the expression to 4 significant figures."""
        # Use regex to find floating point numbers in the expression
        def round_float_match(match):
            full_match = match.group(0)
            try:
                # Parse the float value
                value = float(full_match)
                # Round to 4 significant figures
                if value != 0:
                    return f"{value:.4g}"
                return "0.0"
            except ValueError:
                # If not a valid float, return the original string
                return full_match
                
        # Pattern to match floating point numbers with word boundaries
        # This ensures we don't match numbers that are part of variable names
        float_pattern = r'\b[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\b'
        # Replace all floats with rounded versions
        return re.sub(float_pattern, round_float_match, expr)
     
    def _attempt_llm_simplification(self, best_expression, x_data, y_data, Ninputs, ranges,
                                   client_to_use, gpt_model, custom_system_prompt, 
                                   num_answers_per_prompt, timeout_simplify, prune_amount, 
                                   attempt_num, number_of_prompts = 3,  timeout_regular = 5):
        """Attempt to simplify an expression using LLM."""
        try:
            # Round floats in best_expression to 4 significant figures
            if isinstance(best_expression, str):
                best_expression = self._round_floats_in_expression(best_expression)

            self.logger.info(f"LLM simplification step - trying {num_answers_per_prompt} answers per prompt for {number_of_prompts} prompts. We are on attempt #{attempt_num+1}")
            expr_llm_list = self._call_model_simplify(
                ranges, best_expression, client=client_to_use, gpt_model=gpt_model, 
                system_prompt=custom_system_prompt, 
                sympy=True, numpy=False, 
                num_answers=num_answers_per_prompt, number_of_prompts=number_of_prompts
            )
            self.logger.info(f"LLM improvement responses ({num_answers_per_prompt} answers per prompt for {number_of_prompts} prompts, {len(expr_llm_list)} responses, using {num_answers_per_prompt*number_of_prompts}): {expr_llm_list}")
            
            # Initialize best LLM expression result
            best_llm_result = {
                'expression': None,
                'expression_numpy': None,
                'n_chi_squared': float('inf'),
                'fit_type': 'LLMsimplified',
                'original_llm_expr': None
            }
            
            # List to collect all valid LLM expressions
            all_llm_results = []
            
            # Try each of the LLM simplified expressions
            for answer_num, expr_llm in enumerate(expr_llm_list[:num_answers_per_prompt*number_of_prompts], 1): #answer_num is now indexed from 0, but we still get the first element
                try:
                    if not expr_llm.strip():
                        self.logger.warning(f"Empty LLM expression, skipping")
                        continue
                    self.logger.info(f"Trying LLM simplified expression #{answer_num}: {self._round_floats_in_expression(expr_llm)}")
                    expr_llm_numpy = self._convert_sympy_to_numpy(expr_llm)
                    
                    # First evaluate the n_chi-squared of the LLM simplified expression without fitting
                    try:
                        f_llm, _ = self._convert_to_np_function(expr_llm_numpy, Ninputs, params=True, return_string=True)
                        n_chi_squared_no_fit = get_n_chi_squared(
                            x_data, y_data, f_llm, [], 
                            explain_if_inf=True, string_for_explanation=expr_llm_numpy
                        )
                        self.logger.info(f"LLM prompt number #{answer_num} n_chi2 without fitting: {n_chi_squared_no_fit:.4e}")
                        
                        # Try to fit parameters
                        curve_ansatz_str_np, params_initial = self._replace_floats_with_params(expr_llm_numpy)
                        curve_np, curve_ansatz_np = self._convert_to_np_function(curve_ansatz_str_np, Ninputs, params=True, return_string=True)
                        
                        try:
                            params_opt, n_chi_squared_after_fitting = self._fit_params(
                                x_data, y_data, curve_np, params_initial, curve_ansatz_np, 
                                log_methods=True, log_everything=True, timeout_regular=timeout_regular
                            )
                            self.logger.info(f"LLM prompt number #{answer_num} n_chi2 after fitting: {n_chi_squared_after_fitting:.4e} for {curve_ansatz_np}, params opt.: {str(params_opt[:3])[:-1]}...")
                            n_chi_squared_after_llm_fitting = n_chi_squared_after_fitting
                        except Exception as fit_error:
                            self.logger.warning(f"Fitting failed for LLM expression #{answer_num}: {fit_error}")
                            # Use the n_chi-squared without fitting if fitting fails
                            n_chi_squared_after_llm_fitting = n_chi_squared_no_fit
                            params_opt = params_initial
                        
                        # Always add this result to our list of all LLM results
                        params_opt = self._prune_small_params(params_opt, prune_amount)
                        current_llm_expr = self._simplify_expression(
                            self._subst_params(curve_ansatz_str_np, params_opt, round_to=16), 
                            Ninputs, timeout=timeout_simplify
                        )
                        current_llm_expr_numpy = self._convert_sympy_to_numpy(current_llm_expr)
                        
                        current_result = {
                            'expression': current_llm_expr,
                            'expression_numpy': current_llm_expr_numpy,
                            'n_chi_squared': n_chi_squared_after_llm_fitting,
                            'fit_type': 'LLMsimplified',
                            'original_llm_expr': expr_llm,
                            'answer_num': answer_num,
                        }
                        all_llm_results.append(current_result)
                        
                        # Update best LLM expression if this one is better
                        if n_chi_squared_after_llm_fitting < best_llm_result['n_chi_squared']:
                            best_llm_result = current_result.copy()
                            
                            # If we found an excellent fit, break out early
                            if n_chi_squared_after_llm_fitting < 1e-6:
                                self.logger.info(f"Found excellent fit with n_chi_squared < 1e-6: {n_chi_squared_after_llm_fitting:.4e}")
                                break
                    except Exception as e3:
                        self.logger.warning(f"Error evaluating n_chi-squared for LLM expression #{answer_num}: {e3}")
                except Exception as e2:
                    self.logger.warning(f"Error with LLM expression #{answer_num}: {e2}")
                
            return best_llm_result, all_llm_results
            
        except Exception as e:
            self.logger.warning(f"Error in LLM simplification loop, may try again - only a few times. Current attempt number: #{attempt_num+1}: {e}")
            return {
                'expression': None,
                'expression_numpy': None,
                'n_chi_squared': float('inf'),
                'fit_type': 'LLMsimplified',
                'original_llm_expr': None
            }, []

    def _process_expression_using_llm(self, best_expression, x_data, y_data, Ninputs, ranges,
                                         client, gpt_model, custom_system_prompt, num_answers_per_prompt,
                                         timeout_simplify, number_of_prompts, prune_small_terms):
        """Process and evaluate LLM simplified expressions."""
        try:
            # Use provided client or instance client
            client_to_use = client if client is not None else self.client
            if client_to_use is None:
                raise ValueError("Client must be provided either during initialization or to this method call")
            
            # Update instance client
            self.client = client_to_use
            
            # Set pruning amount
            prune_amount = 1e-6 if prune_small_terms is True else prune_small_terms
            
            # Initialize best LLM expression result
            best_llm_result = {
                'expression': None,
                'expression_numpy': None,
                'n_chi_squared': float('inf'),
                'fit_type': 'LLMsimplified',
                'original_llm_expr': None
            }
            
            # List to collect all LLM results
            all_llm_results = []
            
            # Try multiple prompts to get a valid LLM expression
            number_of_attempts = 2
            self.logger.info(f"LLM simplification step - trying {number_of_prompts} prompts, each with {num_answers_per_prompt} answers. Will attempt this process {number_of_attempts} times.")
            for attempt_num in range(number_of_attempts):
                attempt_result, attempt_all_results = self._attempt_llm_simplification(
                    best_expression, x_data, y_data, Ninputs, ranges,
                    client_to_use, gpt_model, custom_system_prompt, 
                    num_answers_per_prompt, timeout_simplify, prune_amount, 
                    attempt_num, number_of_prompts = number_of_prompts
                )
                
                # Add all valid results to our collection
                all_llm_results.extend(attempt_all_results)
                
                # If we got a valid result, update best result
                if isinstance(attempt_result, dict) and attempt_result['expression'] is not None:
                    if attempt_result['n_chi_squared'] < best_llm_result['n_chi_squared']:
                        best_llm_result = attempt_result
                
                # If we found a reasonable fit, break out of the multiple attempts
                if best_llm_result['n_chi_squared'] < 100:
                    break
                
            # Log final LLM result
            if best_llm_result['expression'] is not None:
                self.logger.info(f"Final LLM response, n_chi2 {best_llm_result['n_chi_squared']:.4e} simplified and refitted: {self._round_floats_in_expression(best_llm_result['expression'])}, from model response {best_llm_result['original_llm_expr']}")
            else:
                self.logger.warning("All LLM simplifications failed to fit properly")
            
            return best_llm_result, all_llm_results
            
        except Exception as e:
            self.logger.warning(f"Skipping LLM improvement: {e}")
            return {
                'expression': None,
                'expression_numpy': None,
                'n_chi_squared': float('inf'),
                'fit_type': 'LLMsimplified',
                'original_llm_expr': None
            }, []

    def _create_result_dict(self, raw_result, refitted_result, llm_result, best_result):
        """Create a comprehensive result dictionary for a single expression."""
        # Ensure consistent handling of results that might be single values or lists
        def extract_value(result, key):
            if result is None or result[key] is None:
                return None
            return result[key]
            
        return {
            'raw_expression': extract_value(raw_result, 'expression_numpy'),
            'raw_n_chi_squared': extract_value(raw_result, 'n_chi_squared'),
            'final_refitted_expression': extract_value(refitted_result, 'expression_numpy'),
            'n_chi_squared_refitted': extract_value(refitted_result, 'n_chi_squared'),
            'final_LLM_expression': extract_value(llm_result, 'expression_numpy'),
            'n_chi_squared_LLM_final': extract_value(llm_result, 'n_chi_squared'),
            'best_expression': extract_value(best_result, 'expression_numpy'),
            'best_n_chi_squared': extract_value(best_result, 'n_chi_squared'),
            'best_fit_type': extract_value(best_result, 'fit_type')
        }

    def optimise_expressions(self, client=None, simplification_gpt_model="openai/gpt-4o", x_data=None, y_data=None, custom_system_prompt=None, 
                            prune_small_terms=True, plot_all=True, original_f=None,
                            num_answers_per_prompt=3, timeout_simplify=10, number_of_prompts=3, print_top_n=10):
        """
        Optimise and simplify the final expressions.

        Args:
            simplification_gpt_model: GPT model to use for simplification
            x_data: x data points (if None, uses training data)
            y_data: y data points (if None, uses training data)
            custom_system_prompt: Custom system prompt for LLM
            prune_small_terms: Whether to prune small terms
            plot_all: Whether to plot results after optimisation
            original_f: Original function for comparison (optional)
            num_answers_per_prompt: Number of answers per prompt
            timeout_simplify: Timeout for simplification in seconds
            number_of_prompts: Number of prompts to try
            print_top_n: Number of top fits to print
        Returns:
            Tuple of (best_expressions, best_n_chi_squareds, detailed_results, all_results_by_output)
            where all_results_by_output is a list where each element corresponds to a KAN output,
            containing all results for that output sorted by n_chi_squared (best to worst)
        """
        # Validate inputs and prepare data
        self._validate_optimisation_prerequisites(x_data, y_data)
        x_data, y_data, full_expressions, Ninputs, ranges = self._prepare_optimisation_data(x_data, y_data)
        
        # Setup for processing results
        results_all_dicts = []
        
        # Dictionary to collect all results grouped by output index
        all_results_by_output = {i: [] for i in range(len(full_expressions))}
        
        # Process each expression
        for i, expr in enumerate(full_expressions):
            self.logger.info(f"\n###################################################")
            self.logger.info(f"Simplifying output {i} using {simplification_gpt_model} and sympy")
            
            # Track best results
            best_result = {
                'n_chi_squared': float('inf'),
                'expression': None,
                'expression_numpy': None,
                'fit_type': None
            }
            
            # Process raw expression
            raw_result = self._process_raw_expression(expr, x_data, y_data, Ninputs, ranges)
            self._update_best_result(best_result, raw_result)
            
            # Add raw result to all_results if it has a valid expression
            if raw_result['expression'] is not None:
                result_entry = {
                    'expression': raw_result['expression'],
                    'expression_numpy': raw_result['expression_numpy'],
                    'n_chi_squared': raw_result['n_chi_squared'],
                    'fit_type': raw_result['fit_type'],
                    'output_index': i
                }
                all_results_by_output[i].append(result_entry)
           
            # Process KAN simplified expression
            refitted_result = self._process_and_refit_raw_expression(
                expr, x_data, y_data, Ninputs, 
                timeout_simplify, prune_small_terms, raw_result
            )
            self._update_best_result(best_result, refitted_result)
            
            # Add KAN result to all_results if it has a valid expression
            if refitted_result['expression'] is not None:
                result_entry = {
                    'expression': refitted_result['expression'],
                    'expression_numpy': refitted_result['expression_numpy'],
                    'n_chi_squared': refitted_result['n_chi_squared'],
                    'fit_type': refitted_result['fit_type'],
                    'output_index': i
                }
                all_results_by_output[i].append(result_entry)

            self.logger.info("Current best result:")
            self.logger.info(f"fit_type: {best_result['fit_type']}")
            self.logger.info(f"n_chi_squared: {best_result['n_chi_squared']}")
            self.logger.info(f"expression: {best_result['expression']}")
            
            # Process LLM simplified expression
            llm_result, all_llm_results = self._process_expression_using_llm(
                best_result['expression'], x_data, y_data, Ninputs, ranges, 
                client, simplification_gpt_model, custom_system_prompt, num_answers_per_prompt,
                timeout_simplify, number_of_prompts, prune_small_terms
            )
            self._update_best_result(best_result, llm_result)
            
            # Add best LLM result to all_results if it has a valid expression
            if llm_result['expression'] is not None:
                result_entry = {
                    'expression': llm_result['expression'],
                    'expression_numpy': llm_result['expression_numpy'],
                    'n_chi_squared': llm_result['n_chi_squared'],
                    'fit_type': llm_result['fit_type'],
                    'output_index': i
                }
                all_results_by_output[i].append(result_entry)
            
            # Add all other LLM results to all_results
            for llm_res in all_llm_results:
                if llm_res['expression'] is not None and llm_res != llm_result:
                    result_entry = {
                        'expression': llm_res['expression'],
                        'expression_numpy': llm_res['expression_numpy'],
                        'n_chi_squared': llm_res['n_chi_squared'],
                        'fit_type': f"{llm_res['fit_type']}_attempt_{llm_res.get('attempt_num', 0)}_answer_{llm_res.get('answer_num', 0)}",
                        'output_index': i,
                        'original_llm_expr': llm_res.get('original_llm_expr', None)
                    }
                    all_results_by_output[i].append(result_entry)
            
            # Store results for this expression
            result_dict = self._create_result_dict(raw_result, refitted_result, llm_result, best_result)
            results_all_dicts.append(result_dict)
        
        # Extract final results
        best_expressions = [result_dict['best_expression'] for result_dict in results_all_dicts]
        best_n_chi_squareds = [result_dict['best_n_chi_squared'] for result_dict in results_all_dicts]
        
        #Sort results for each output by n_chi_squared (best to worst)
        all_results_sorted = []
        for i in range(len(full_expressions)):
            output_results = sorted(all_results_by_output[i], key=lambda x: x['n_chi_squared'])
            all_results_sorted.append(output_results)
            
        # Now do all the printing and plotting at the end
        for i, expr in enumerate(full_expressions):
            raw_result = next((r for r in all_results_by_output[i] if r['fit_type'] == 'raw'), None)
            refitted_result = next((r for r in all_results_by_output[i] if r['fit_type'] == 'rawrefitted'), None)
            llm_result = next((r for r in all_results_by_output[i] if r['fit_type'] == 'LLMsimplified'), None)
            best_result = results_all_dicts[i]
            
            # Plot if requested
            if plot_all and Ninputs == 1:
                # Define xs for plotting
                xs = np.linspace(ranges[0][0], ranges[0][1], 100)

                # Setup for plotting if requested
                fig, ax, xs = self._setup_plotting(plot_all, ranges, Ninputs, original_f, xs=xs)
                
                # Only plot results that exist
                if raw_result is not None:
                    self._plot_expression(ax, xs, raw_result, "KAN_LEx (raw)")
                if refitted_result is not None:
                    self._plot_expression(ax, xs, refitted_result, "KAN_LEx (simp. and refit.)")
                if llm_result is not None:
                    self._plot_expression(ax, xs, llm_result, "KAN_LEx (after LLM simp. and refit.)")
                
                ax.legend()
                plt.show()
            elif plot_all and Ninputs > 1:
                self.logger.error(f"Plotting all results for output {i} with {Ninputs} inputs is not supported for more than 1 input. Skipping plot. If Ninputs ==2, use self.plot_results()")
                
            # Print raw and refitted results
            self.logger.info(f"\n###############################\n# Raw and Refitted Results for output {i} (4sf): #\n###############################")
            
            # Only print results that exist
            if raw_result is not None:
                self.logger.info(f"Raw expression n_chi2 {raw_result['n_chi_squared']:.3e}: {self._round_floats_in_expression(raw_result['expression'])}")
            else:
                self.logger.info("Raw expression: None")
                
            if refitted_result is not None:
                self.logger.info(f"Refitted expression n_chi2 {refitted_result['n_chi_squared']:.3e}: {self._round_floats_in_expression(refitted_result['expression'])}")
            else:
                self.logger.info("Refitted expression: None")
                
            if llm_result is not None:
                self.logger.info(f"Best LLM expression n_chi2 {llm_result['n_chi_squared']:.3e}: {self._round_floats_in_expression(llm_result['expression'])}")
            else:
                self.logger.info("Best LLM expression: None")
            
            # Log final results
            self.logger.info(f"\n###############################\n# Final formula for output {i}: #\n###############################")
            self.logger.info(f"Best expression n_chi2 {best_result['best_n_chi_squared']:.3e} from {best_result['best_fit_type']} fit: {best_result['best_expression']}")
            self.logger.info(f"Round to 4sf: {self._round_floats_in_expression(best_result['best_expression'])}")
            
            # Print top N fits for this specific output
            self.logger.info(f"\n###############################\n# Top {print_top_n} Best Fits for Output {i} #\n###############################")
            self._print_top_n_fits_for_output(all_results_sorted[i], print_top_n)
        
        # # Print top-N best fits across all outputs
        # self.logger.info("\n\n###############################")
        # self.logger.info(f"# Top {print_top_n} Best Fits Overall #")
        # self.logger.info("###############################")
        
        # # Call the method to print top N fits across all outputs
        # self.print_top_10_best_fits(all_results_sorted, print_top_n=print_top_n)
        
        return best_expressions, best_n_chi_squareds, results_all_dicts, all_results_sorted

    def _print_top_n_fits_for_output(self, output_results, print_top_n=10):
        # Sort by n_chi_squared (best to worst)
        results_sorted = sorted(output_results, key=lambda x: x['n_chi_squared'])
        
        # Print top N (or fewer if less than N available)
        print_top_n = min(print_top_n, len(results_sorted))
        
        # Create table header
        self.logger.info(f"{'Chi²':<15} {'Type':<30} {'Expression (rounded to 4sf)'}")
        self.logger.info("-" * 90)
        
        for i in range(print_top_n):
            result = results_sorted[i]
            # Round expression to 4 significant figures
            rounded_expr = self._round_floats_in_expression(result['expression'])
            # Format and print table row
            self.logger.info(f"{result['n_chi_squared']:<15.3e} {result['fit_type']:<30} {rounded_expr}")

    def _convert_to_np_function(self, expr, Ninputs, floats_only=False, params=False, return_string=False):
        """Convert a string expression to a function."""
        # Check if expr has any x# variables in it
        has_variables = any(f"x{i}" in expr for i in range(Ninputs))
        if not has_variables:
            self.logger.warning(f"Expression {expr} does not depend on x variables, adding x0**0 to make it dependent on x0")
            # Add x0**0 (which equals 1) to make the expression depend on x0
            expr = f"({expr})*(x0**0)"
            
        if floats_only:
            lambda_xi = "lambda " + ", ".join([f"x{i}" for i in range(Ninputs)]) + " : " + expr
            func = eval(lambda_xi, {"np": np})
        elif params:
            lambda_xi = "lambda " + ", ".join([f"x{i}" for i in range(Ninputs)]) + ", *params :" + expr 
            func = eval(lambda_xi, {"np": np})
        else:
            raise ValueError("Invalid mode")
        if return_string:
            return func, lambda_xi
        else:
            return func

    def _process_symbolic_expressions_to_python_function(self, x, y_true, y_raw, y_simplified, try_jax = True):
        """Process symbolic expressions to generate and optimise functions."""
        try:
            # Generate function from symbolic expressions
            python_generated_function, learned_func_str, best_params, optimised_params, total_params = self.generate_learned_f_function(optimise_params = True, x = x, y_true = y_true, try_jax = try_jax)
            # Generate predictions with best parameters
            y_generated = python_generated_function(x, *best_params)# x arraylike
            y_generated_optimised = python_generated_function(x, *optimised_params)
            # Verify that generated function matches raw expression
            is_equivalent, message = test_data_equivalence(y_generated, y_raw, rtol=1e-5)
            if not is_equivalent:
                self.logger.warning(f"Python program generated directly from symbolic expressions doesn't match raw expression up to 1e-5. Possible error in symbolic simplification/floating point rounding? Average relative difference: {message}")
            elif message:
                self.logger.info(f"Python program generated directly from symbolic expressions and raw expression: {message}")
            else:
                self.logger.info("Python program generated directly from symbolic expressions and raw expression match up to 1e-5")
            
            # Verify that optimised function matches simplified expression
            is_equivalent, message = test_data_equivalence(y_generated_optimised, y_simplified, rtol=1e-5)
            if not is_equivalent:
                self.logger.warning(f"Numerically optimised python program generated directly from symbolic expressions doesn't match simplified expression. Possible error in symbolic simplification/floating point rounding? Average relative difference: {message}")
            elif message:
                self.logger.info(f"Numerically optimised python program generated directly from symbolic expressions and simplified expression: {message}")
            else:
                self.logger.info("Numerically optimised python program generated directly from symbolic expressions and simplified expression match exactly")
            
            return y_generated, y_generated_optimised
            
        except Exception as e:
            self.logger.warning(f"Error in processing symbolic KAN directly to python function: {e}")
            return None, None

    def generate_learned_f_function(self,optimise_params = True, x = None, y_true = None, try_jax = True,  result_of_kan_to_symbolic = None):
        self.logger.info("Argument x of learned function are arraylike, NOT x0, x1, etc.")
        if result_of_kan_to_symbolic is None:
            result_of_kan_to_symbolic = self.symbolic_expressions
        learned_func_str, total_params, best_params =  LLMLEx.generate_learned_f(result_of_kan_to_symbolic)
        local_vars = {}
        exec(learned_func_str, {"np": np}, local_vars)
        python_generated_function = local_vars.get('learned_f')
        if not optimise_params:
            return python_generated_function, learned_func_str, best_params, total_params
        else:
            optimised_params = self._find_optimised_for_full_python_kan_function(python_generated_function, x, y_true, best_params, learned_func_str, try_jax = try_jax)
            return python_generated_function, learned_func_str, best_params, optimised_params, total_params

    def _find_optimised_for_full_python_kan_function(self, python_generated_function, x, y_true, best_params, learned_func_str, try_jax = True):
        """
        Find optimised parameters for the generated Python function.
        
        Args:
            python_generated_function: The generated Python function
            x: Input data
            y_true: True output data
            best_params: Initial parameters
            learned_func_str: String representation of the learned function
            local_vars: Dictionary of local variables
            
        Returns:
            Tuple of (optimised_params, y_generated_optimised)
        """
        # Try both scipy and jax optimisers and use whichever works best
        scipy_success = False
        jax_success = False
        best_mse = float('inf')
        best_params = best_params  # Start with initial params
        best_y_generated = None
        
        # First try scipy
        try:
            from scipy.optimize import minimize
            def objective(params):
                predictions = python_generated_function(x, *params)
                return np.mean((predictions - y_true)**2)
            
            optimised_result = minimize(objective, best_params)
            scipy_params = optimised_result.x
            scipy_y_generated = python_generated_function(x, *scipy_params)
            scipy_mse = np.mean((scipy_y_generated - y_true)**2)
            
            scipy_success = True
            best_mse = scipy_mse
            best_params = scipy_params
            best_y_generated = scipy_y_generated
            self.logger.info(f"SciPy optimisation successful with MSE: {scipy_mse}")
        except Exception as e:
            self.logger.warning(f"SciPy optimisation failed: {e}")
        
        # Then try JAX
        if try_jax:
            try:
                import jax.numpy as jnp
                from jax.scipy.optimize import minimize as jax_minimize

                learned_func_str_jax = learned_func_str.replace('np.', 'jnp.')
                local_vars = {} 
                exec(learned_func_str_jax, {"jnp": jnp}, local_vars)
                python_generated_function_jax = local_vars.get('learned_f')

                def jax_objective(params):
                    predictions = python_generated_function_jax(jnp.array(x), *params)
                    return jnp.mean((predictions - jnp.array(y_true))**2)

                params_initial = jnp.array(best_params)
                result = jax_minimize(jax_objective, params_initial)

                jax_params = result.x
                jax_y_generated = python_generated_function(x, *jax_params)
                jax_mse = np.mean((jax_y_generated - y_true)**2)

                jax_success = True
                if jax_mse < best_mse:
                    best_mse = jax_mse
                    best_params = jax_params
                    best_y_generated = jax_y_generated
                    self.logger.info(f"JAX optimisation improved results with MSE: {jax_mse}")
                else:
                    self.logger.info(f"JAX optimisation completed but didn't improve over SciPy (MSE: {jax_mse})")
            except Exception as e:
                self.logger.warning(f"JAX optimisation failed: {e}")
        
        if not scipy_success and not jax_success:
            self.logger.warning("Both optimisation methods failed, using original parameters")
        
        return best_params

    def plot_results(self, ranges=None, result_dict=None, dataset=None, title="KAN Symbolic Regression Results", 
                    plotmaxmin=[[None, None], [None, None]], plot_using_generate_f = False, n_points_in_each_direction=50, wireframe_fit=True,  plot_points=True, expressions_to_plot='best'):
        """
        Plot the original function and the approximations.
        
        Args:
            ranges: Tuple of (min_x, max_x) for the input range. If None, will try to derive from dataset.
            result_dict: Dictionary with results from optimise_expressions. If None, will use self.results_all_dicts[0] if available.
            dataset: Optional dataset to use for plotting. If None, will use self.dataset if available.
            title: Plot title
            plotmaxmin: Limits for the plot [[xmin, xmax], [ymin, ymax]]
            plot_using_generate_f: Whether to plot the generated function
        2D only arguments:
            n_points_in_each_direction: Number of points to use in each direction for 2D plots
            wireframe_fit: Whether to plot the best expression as a wireframe or surface
            plot_all_expressions: Whether to plot all expressions (only for 1D)
            plot_points: Whether to plot the data points (only for 2D)
            expressions_to_plot: Which expression to plot for 2D case. Options: 'best', 'raw', 'simplified_refitted', 
                               'simplified_by_llm', 'generated', 'generated_optimised'. or a list of these.
            
        Returns:
            matplotlib figure and axes objects
        """
        if expressions_to_plot is None:
            expressions_to_plot = []
        if isinstance(expressions_to_plot, str):
            expressions_to_plot = [expressions_to_plot]
        # Ensure we have a valid result_dict
        if result_dict is None:
            if hasattr(self, 'results_all_dicts') and self.results_all_dicts:
                result_dict = self.results_all_dicts[0]
            else:
                raise ValueError("result_dict must be provided or self.results_all_dicts must be set")
        
        # Use provided dataset or fall back to self.dataset
        if dataset is None and hasattr(self, 'dataset') and self.dataset is not None:
            dataset = self.dataset
            self.logger.info("Using internal dataset for plotting")
        
        # Determine ranges
        if ranges is None:
            # Try to derive ranges from dataset
            if dataset is not None:
                # Extract ranges from the dataset
                try:
                    x_data = dataset['train_input'].cpu().numpy()
                    Ninputs = x_data.shape[-1] if len(x_data.shape) > 1 else 1
                    if len(x_data.squeeze().shape) > 1:
                        Ninputs = x_data.shape[-1]
                        # Multiple input variables, use first dimension
                        ranges = [(float(np.min(x_data[:, i])), float(np.max(x_data[:, i]))) for i in range(Ninputs)]
                    else:
                        Ninputs = 1
                        ranges = (float(np.min(x_data)), float(np.max(x_data)))
                    self.logger.info(f"Using ranges derived from dataset: {ranges}")
                except Exception as e:
                    self.logger.warning(f"Could not derive ranges from dataset: {e}")
                    
            # If still no ranges, use default
            if ranges is None:
                ranges = (-np.pi, np.pi)
                self.logger.warning(f"No ranges provided or derivable from dataset. Using default: {ranges}")
        else:
            Ninputs = len(ranges) if isinstance(ranges, list) and all(isinstance(r, (list, tuple)) for r in ranges) else 1
            self.logger.info(f"Using provided ranges: {ranges}")
        if Ninputs == 1:
            # Ensure ranges is a tuple of two values
            if not isinstance(ranges, (list, tuple)) or len(ranges) != 2:
                raise ValueError("ranges must be a tuple of (min_x, max_x)")

            x = np.linspace(ranges[0], ranges[1], 1000)

            # Get true values from various sources
            y_true = None

            # First option: use dataset if available
            if dataset is not None:
                try:
                    # Try to use the training data for the true values
                    train_input = dataset['train_input'].cpu().numpy()
                    train_label = dataset['train_label'].cpu().numpy()

                    # If x values in the dataset exactly match our linspace, use directly
                    if len(train_input) == 1000 and np.allclose(train_input.flatten(), x):
                        y_true = train_label.flatten()
                    else:
                        # For multivariate data, we can only use the first variable for plotting
                        if len(train_input.shape) > 1 and train_input.shape[1] > 1:
                            self.logger.info("Using first variable for plotting multivariate data")

                    # If we have a function, we'll prefer using that for smooth evaluation
                    if not callable(getattr(self, 'f', None)):
                        self.logger.info("Using dataset for ground truth rather than evaluating function")
                except Exception as e:
                    self.logger.warning(f"Error using dataset for true values: {e}")

            # Second option: use self.f if available and the dataset method failed
            if y_true is None and hasattr(self, 'f') and callable(self.f):
                try:
                    self.logger.info("Using self.f function for ground truth")
                    # Try different input formats for the function
                    try:
                        y_true = self.f(torch.tensor(x).reshape(-1, 1).float()).numpy().flatten()
                    except (TypeError, AttributeError):
                        try:
                            y_true = self.f(x)
                            if isinstance(y_true, torch.Tensor):
                                y_true = y_true.numpy()
                        except (TypeError, AttributeError):
                            y_true = np.array([self.f(xi) for xi in x])
                except Exception as e:
                    self.logger.warning(f"Error computing true function values from self.f: {e}")
                    y_true = None

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the true function if we have it
            if y_true is not None:
                ax.plot(x, y_true, 'b-', label='True Function', linewidth=4, alpha=0.5)

            # Get the best expression
            best_expr = result_dict['best_expression']
            best_n_chi_squared = result_dict['best_n_chi_squared']

            # Try to plot the best expression
            try:
                y_best = eval(best_expr, {"np": np, "x0": x})
                ax.plot(x, y_best, 'r--', label=f'Best Expression (χ²={best_n_chi_squared:.5e})', linewidth=2)
            except Exception as e:
                self.logger.warning(f"Error plotting best expression: {e}")

            raw_expr = result_dict['raw_expression']
            simplified_refitted_expr = result_dict['final_refitted_expression']
            simplified_by_LLM_expr = result_dict['final_LLM_expression']
            # Try to plot the simplified expression
            try:
                if raw_expr is not None:
                    raw_n_chi_squared = result_dict['raw_n_chi_squared']
                    y_raw= self._convert_to_np_function(raw_expr, Ninputs, floats_only=True)(x)
                    self.logger.info('Plotting raw expression')
                    ax.plot(x, y_raw, 'orange', dashes=[4, 2], label=f'Raw expression from LLMLEx (χ²={raw_n_chi_squared:.5e})', linewidth=2)
                if simplified_refitted_expr is not None:
                    n_chi_squared = result_dict['n_chi_squared_refitted']
                    y_simplified = self._convert_to_np_function(simplified_refitted_expr, Ninputs, floats_only=True)(x)
                    self.logger.info('Plotting simplified and refitted/pruned raw expression')
                    ax.plot(x, y_simplified, 'g-.', dashes=[3, 1, 1, 1], label=f'Simplified and refitted/pruned raw expression (χ²={n_chi_squared:.5e})', linewidth=2)
                if simplified_by_LLM_expr is not None:
                    n_chi_squared_LLM = result_dict['n_chi_squared_LLM_final']
                    y_simplified_LLM = self._convert_to_np_function(simplified_by_LLM_expr, Ninputs, floats_only=True)(x)
                    self.logger.info('Plotting simplified by LLM expression')
                    ax.plot(x, y_simplified_LLM, 'c--', dashes=[2, 1], label=f'Simplified by LLM and refitted/pruned (χ²={n_chi_squared_LLM:.5e})', linewidth=2)
            except Exception as e:
                self.logger.warning(f"Error plotting one of the expressions: {e}, simplified by LLM expression: {self._round_floats_in_expression(simplified_by_LLM_expr)}, refitted /pruned raw expression: {self._round_floats_in_expression(simplified_refitted_expr)}")
            if self.symbolic_expressions is not None:
                try:
                    # Call the method to process symbolic expressions
                    y_generated, y_generated_optimised = self._process_symbolic_expressions_to_python_function(x, y_true, y_raw, y_simplified)
                    # Plot if requested
                    if plot_using_generate_f:
                        final_n_chi_squared = get_n_chi_squared_from_predictions(x, y_true, y_generated)
                        final_n_chi_squared_optimised = get_n_chi_squared_from_predictions(x, y_true, y_generated_optimised)
                        ax.plot(x, y_generated, 'k-', label=f'Raw python program (χ²={final_n_chi_squared:.5e})', linewidth=4, alpha=0.3)
                        ax.plot(x, y_generated_optimised, 'k--', label=f'Optimised python program (χ²={final_n_chi_squared_optimised:.5e})', linewidth=4, alpha=0.3)
                except Exception as e:
                    self.logger.warning(f"Error processing generated function: {e}")
            # Try to plot the model and pruned model predictions if they're part of this instance
            # (for tests, we may not have actual models)
            if y_true is not None:  # Only try to calculate chi-squared if we have true values
                try:
                    if hasattr(self, 'raw_model') and self.raw_model is not None:
                        try:
                            model_preds = self.raw_model(torch.tensor(x).reshape(-1, 1).float()).detach().numpy().flatten()
                            try:
                                n_chi_squared = get_n_chi_squared_from_predictions(x, y_true, model_preds)
                                ax.plot(x, model_preds, 'b:', dashes=[1, 1], label=f'KAN Model {n_chi_squared:.5e}', linewidth=2)
                            except (ImportError, NameError):
                                # Fall back if the function isn't available
                                ax.plot(x, model_preds, 'b:', dashes=[1, 1], label=f'KAN Model', linewidth=2)
                        except Exception as e:
                            self.logger.warning(f"Error plotting KAN model: {e}")

                    if hasattr(self, 'model') and self.model is not None and self.model != getattr(self, 'raw_model', None):
                        try:
                            pruned_preds = self.model(torch.tensor(x).reshape(-1, 1).float()).detach().numpy().flatten()
                            try:
                                n_chi_squared = get_n_chi_squared_from_predictions(x, y_true, pruned_preds)
                                ax.plot(x, pruned_preds, 'm:', dashes=[3, 1], label=f'Pruned KAN Model {n_chi_squared:.5e}', linewidth=2)
                            except (ImportError, NameError):
                                # Fall back if the function isn't available
                                ax.plot(x, pruned_preds, 'm:', dashes=[3, 1], label=f'Pruned KAN Model', linewidth=2)
                        except Exception as e:
                            self.logger.warning(f"Error plotting pruned KAN model: {e}")
                except Exception as e:
                    self.logger.warning(f"Error plotting KAN models: {e}")
            else:
                # If we don't have true values but still want to plot the models
                try:
                    if hasattr(self, 'raw_model') and self.raw_model is not None:
                        try:
                            model_preds = self.raw_model(torch.tensor(x).reshape(-1, 1).float()).detach().numpy().flatten()
                            ax.plot(x, model_preds, 'b:', dashes=[1, 1], label=f'KAN Model', linewidth=2)
                        except Exception as e:
                            self.logger.warning(f"Error plotting KAN model: {e}")

                    if hasattr(self, 'model') and self.model is not None and self.model != getattr(self, 'raw_model', None):
                        try:
                            pruned_preds = self.model(torch.tensor(x).reshape(-1, 1).float()).detach().numpy().flatten()
                            ax.plot(x, pruned_preds, 'm:', dashes=[3, 1], label=f'Pruned KAN Model', linewidth=2)
                        except Exception as e:
                            self.logger.warning(f"Error plotting pruned KAN model: {e}")
                except Exception as e:
                    self.logger.warning(f"Error plotting KAN models: {e}")

            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Apply limits if specified
            if plotmaxmin[0][0] is not None:
                ax.set_xlim(left=plotmaxmin[0][0])
            if plotmaxmin[0][1] is not None:
                ax.set_xlim(right=plotmaxmin[0][1])
            if plotmaxmin[1][0] is not None:
                ax.set_ylim(bottom=plotmaxmin[1][0])
            if plotmaxmin[1][1] is not None:
                ax.set_ylim(top=plotmaxmin[1][1])
        elif Ninputs ==2:
            # Handle bivariate case (2 input variables)
            self.logger.info("Plotting bivariate function")
            
            # Create figure and 3D axis
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create meshgrid for plotting
            x0_points = np.linspace(ranges[0][0], ranges[0][1], n_points_in_each_direction)
            x1_points = np.linspace(ranges[1][0], ranges[1][1], n_points_in_each_direction)
            X0, X1 = np.meshgrid(x0_points, x1_points)
            
            # Get true values if available
            Z_true = None
            if hasattr(self, 'f') and callable(self.f):
                try:
                    self.logger.info("Using self.f function for ground truth")
                    # Try to evaluate the true function on the grid
                    inputs = torch.tensor(np.vstack([X0.flatten(), X1.flatten()]).T).float()
                    Z_true = self.f(inputs).detach().numpy().reshape(X0.shape)
                except Exception as e:
                    self.logger.warning(f"Error computing true function values from self.f: {e}")
            
            # Get test data points if available
            test_data = None
            if dataset is not None:
                try:
                    test_x = dataset['train_input'].cpu().numpy()
                    test_y = dataset['train_label'].cpu().numpy()
                    if len(test_y.squeeze().shape) > 1:
                        self.logger.error("Test data has more than one dimension - not supported for plotting. Returning empty plot.")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        return fig, ax
                    else:
                        test_y = test_y.squeeze()
                    test_x0 = test_x[:, 0]
                    test_x1 = test_x[:, 1]
                    test_data = (test_x0, test_x1, test_y)
                except Exception as e:
                    self.logger.warning(f"Error extracting test data from dataset: {e}")
            
            # Plot the true function surface if available
            if Z_true is not None:
                surf = ax.plot_surface(X0, X1, Z_true, cmap='viridis', alpha=0.5, 
                                      linewidth=0, antialiased=True, label='True Function')
            
            # Get the best expression
            best_expr = result_dict['best_expression']
            best_n_chi_squared = result_dict['best_n_chi_squared']
            
            # Try to plot the best expression
            try:
                best_func = self._convert_to_np_function(best_expr, Ninputs, floats_only=True)
                # Vectorize the function evaluation
                Z_best = np.vectorize(lambda x, y: best_func(x, y))(X0, X1)
                if 'best' in expressions_to_plot:
                    if wireframe_fit:
                        surf_best = ax.plot_surface(X0, X1, Z_best, cmap='plasma', alpha=0.7,
                                              linewidth=0, antialiased=True, label=f"Best Expression (χ²={best_n_chi_squared:.5e})")
                    else:
                        ax.plot_wireframe(X0, X1, Z_best, color='black', alpha=0.5, 
                                     linewidth=1, label=f"Best Expression (χ²={best_n_chi_squared:.5e})")
                ax.set_title(f"{title}\nBest Expression (χ²={best_n_chi_squared:.5e})")
            except Exception as e:
                self.logger.warning(f"Error plotting best expression: {e}")
            
            # Try to plot other expressions if available
            raw_expr = result_dict.get('raw_expression')
            raw_n_chi_squared = result_dict.get('raw_n_chi_squared')
            simplified_refitted_expr = result_dict.get('final_refitted_expression')
            simplified_refitted_n_chi_squared = result_dict.get('final_refitted_n_chi_squared')
            simplified_by_LLM_expr = result_dict.get('final_LLM_expression')
            simplified_by_LLM_n_chi_squared = result_dict.get('final_LLM_n_chi_squared')
            
            # Define expressions to plot with their properties
            expressions_to_plot = [
                {'expr': raw_expr, 'n_chi_squared': raw_n_chi_squared, 'name': 'raw', 'color': 'orange', 'label': 'Raw Expression'},
                {'expr': simplified_refitted_expr, 'n_chi_squared': simplified_refitted_n_chi_squared, 'name': 'final_refitted', 'color': 'green', 'label': 'Simplified & Refitted'},
                {'expr': simplified_by_LLM_expr, 'n_chi_squared': simplified_by_LLM_n_chi_squared, 'name': 'final_LLM', 'color': 'blue', 'label': 'LLM Simplified'}
            ]
            Z_expr_all = {}
            
            # Plot each expression in a loop
            for expr_info in expressions_to_plot:
                expr = expr_info['expr']
                name = expr_info['name']    
                n_chi_squared = expr_info['n_chi_squared']
                func = self._convert_to_np_function(expr, Ninputs, floats_only=True)
                Z_expr_all[name] = np.vectorize(lambda x, y: func(x, y))(X0, X1)
                if expr is not None and (name in expressions_to_plot):
                    try:
                        self.logger.info(f"Plotting {expr_info['name']}")
                        # Vectorize the function evaluation
                        # Plot as wireframe to distinguish from other surfaces
                        if wireframe_fit:
                            ax.plot_wireframe(X0, X1, Z_expr_all[name], color=expr_info['color'], alpha=0.5, 
                                             linewidth=1, label=f"{expr_info['label']} (χ²={n_chi_squared:.5e})" if n_chi_squared is not None else expr_info['label'])
                        else:
                            ax.plot_surface(X0, X1, Z_expr_all[name], color=expr_info['color'], alpha=0.5, 
                                            linewidth=1, label=f"{expr_info['label']} (χ²={n_chi_squared:.5e})" if n_chi_squared is not None else expr_info['label'])
                    except Exception as e:
                        self.logger.warning(f"Error plotting {name}: {e}")
            # Plot the generated function if available
            if self.symbolic_expressions is not None:
                try:
                    # Use the meshgrid directly for processing symbolic expressions
                    x_combined = np.stack((X0, X1), axis=-1)
                    y_true_flat = Z_true.flatten() if Z_true is not None else None
                    
                    # Get raw and simplified expressions if available
                    if 'raw' not in Z_expr_all:
                        self.logger.warning("Raw expression results not available for plotting")
                        y_raw_flat = None
                    else:
                        y_raw_flat = Z_expr_all['raw'].flatten()
                        
                    if 'final_refitted' not in Z_expr_all:
                        self.logger.warning("Simplified expression results not available for plotting")
                        y_simplified_flat = None
                    else:
                        y_simplified_flat = Z_expr_all['final_refitted'].flatten()
                    
                    y_generated, y_generated_optimised = self._process_symbolic_expressions_to_python_function(
                        x_combined, y_true_flat, y_raw_flat, y_simplified_flat)
                    
                    # Check if results are valid before reshaping
                    if y_generated is None:
                        self.logger.warning("Generated function returned None result")
                    else:
                        # Reshape back to grid for plotting
                        Z_generated = y_generated.reshape(X0.shape)
                        
                        if y_generated_optimised is None:
                            self.logger.warning("Optimised generated function returned None result")
                        else:
                            Z_generated_optimised = y_generated_optimised.reshape(X0.shape)
                            
                            # Plot if requested and both results are valid
                            if plot_using_generate_f:
                                self.logger.info("Plotting generated function")
                                if y_true_flat is not None:
                                    final_n_chi_squared = get_n_chi_squared_from_predictions(x_combined, y_true_flat, y_generated)
                                    final_n_chi_squared_optimised = get_n_chi_squared_from_predictions(x_combined, y_true_flat, y_generated_optimised)
                                    raw_label = f'Raw python program (χ²={final_n_chi_squared:.5e})'
                                    opt_label = f'Optimised python program (χ²={final_n_chi_squared_optimised:.5e})'
                                else:
                                    raw_label = 'Raw python program'
                                    opt_label = 'Optimised python program'
                                if 'generated' in expressions_to_plot:
                                    ax.plot_wireframe(X0, X1, Z_generated, color='black', linestyle='-', 
                                                    linewidth=2, alpha=0.3, label=raw_label)
                                if 'generated_optimised' in expressions_to_plot:
                                    ax.plot_wireframe(X0, X1, Z_generated_optimised, color='black', linestyle='--', 
                                                    linewidth=2, alpha=0.3, label=opt_label)
                except Exception as e:
                    self.logger.warning(f"Error plotting generated function: {e}")
                    self.logger.debug(f"Error details:", exc_info=True)
            
            #Plot test data points if available
            if test_data is not None:
                self.logger.info("Plotting test data")
                test_x0, test_x1, test_y = test_data
                print(test_x0.shape, test_x1.shape, test_y.shape)
                ax.scatter(test_x0, test_x1, test_y, c='red', marker='o', s=20, label='Training Data')
            
            #Add labels and customize plot
            ax.set_xlabel('X0', fontsize=12, labelpad=10)
            ax.set_ylabel('X1', fontsize=12, labelpad=10)
            ax.set_zlabel('Y', fontsize=12, labelpad=10)
            
            # Apply limits if specified
            if plotmaxmin[0][0] is not None:
                ax.set_xlim(left=plotmaxmin[0][0])
            if plotmaxmin[0][1] is not None:
                ax.set_xlim(right=plotmaxmin[0][1])
            if plotmaxmin[1][0] is not None:
                ax.set_ylim(bottom=plotmaxmin[1][0])
            if plotmaxmin[1][1] is not None:
                ax.set_ylim(top=plotmaxmin[1][1])
            
            # # Add a legend
            # # Create proxy artists for the legend
            # legend_elements = []
            # if Z_true is not None:
            #     legend_elements.append(plt.Line2D([0], [0], linestyle='none', marker='s', 
            #                                    markerfacecolor='blue', markersize=10, label='True Function'))
            # legend_elements.append(plt.Line2D([0], [0], linestyle='none', marker='s', 
            #                                markerfacecolor='purple', markersize=10, label=f'Best Expression (χ²={best_n_chi_squared:.5e})'))
            # if raw_expr is not None:
            #     legend_elements.append(plt.Line2D([0], [0], color='orange', lw=2, label='Raw Expression'))
            # if test_data is not None:
            #     legend_elements.append(plt.Line2D([0], [0], linestyle='none', marker='o', 
            #                                    markerfacecolor='red', markersize=6, label='Training Data'))
            # if self.symbolic_expressions is not None and plot_using_generate_f:
            #     legend_elements.append(plt.Line2D([0], [0], color='black', lw=2, label='Generated Function'))
            
            ax.legend( loc='upper right', fontsize=10)
            
            # Add a colorbar
            if Z_true is not None:
                cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                cbar.set_label('Function Value', fontsize=12)

        return fig, ax

    def run_complete_pipeline(self, client=None, f=None, ranges=(-np.pi, np.pi), train_steps=50, 
                             generations=3, gpt_model="openai/gpt-4o", node_th=0.2, edge_th=0.2, 
                             custom_system_prompt_for_second_simplification=None, optimiser="LBFGS", 
                             population=10, temperature=0.1, exit_condition=None, verbose=0, 
                             use_async=True, plot_fit=True, plot_parents=False, demonstrate_parent_plotting=False, constant_on_failure=False,
                             simplification_gpt_model=None):
        """
        Run the complete KAN symbolic regression pipeline on a univariate function.
        
        Args:
            client: Client for LLM API calls (defaults to self.client if None)
            f: Target function to approximate
            ranges: Tuple of (min_x, max_x) for the input range
            train_steps: Number of training steps
            generations: Number of generations in the genetic algorithm
            gpt_model: Model to use for simplification
            node_th: Node threshold for pruning
            edge_th: Edge threshold for pruning
            custom_system_prompt_for_second_simplification: Custom system prompt for LLM second simplification
            optimiser: Optimiser to use for training
            population: Population size for genetic algorithm
            temperature: Temperature parameter for the genetic algorithm
            exit_condition: Exit condition for the genetic algorithm
            verbose: Verbosity level
            use_async: Whether to use async execution
            plot_fit: Whether to plot fitting results
            plot_parents: Whether to plot parent solutions
            demonstrate_parent_plotting: Whether to demonstrate parent plotting
            constant_on_failure: Whether to return a constant function on failure
            simplification_gpt_model: GPT model to use for simplification
        Returns:
            Dictionary with complete results including models and expressions
        """
        try:
            # Ensure f is a torch function if provided
            if f is not None:
                if not callable(f):
                    raise ValueError("Target function f must be callable")
                # Check if f is a torch function and not numpy
                import inspect
                source_lines = inspect.getsourcelines(f)[0]
                if not any('torch' in line for line in source_lines):
                    self.logger.error("Target function f must be implemented using torch - double check that it is!")
                if any('numpy' in line for line in source_lines) or any('np.' in line for line in source_lines):
                    raise ValueError("Target function f must not use numpy, use torch instead")
                dataset = self.create_dataset(f, ranges=ranges, n_var=1, train_num=10000, test_num=1000)
            # Use provided client or fallback to the instance's client
            client_to_use = client if client is not None else self.client
            if client_to_use is None:
                raise ValueError("Client must be provided either during initialization or to this method call")
                
            # Use client_to_use for operations that need a client
            self.client = client_to_use  # Update the instance's client
                
            # 1. Create the dataset
            if f is None:
                raise ValueError("Target function f must be provided")
            if simplification_gpt_model is None:
                simplification_gpt_model = gpt_model
                
            
            # 2. Train the model
            self.train_kan(dataset, opt=optimiser, steps=train_steps, prune=True, node_th=node_th, edge_th=edge_th)
            
            self.logger.info("Trained model:")
            self.raw_model.plot()
            self.logger.info("Pruned model:")
            self.model.plot()
            
            # 3. Convert to symbolic expressions
            best_expressions, best_n_chi_squareds, results_all_dicts, all_results_sorted = self.get_symbolic(
                client=client_to_use, population=population, generations=generations, 
                temperature=temperature, gpt_model=gpt_model,
                exit_condition=exit_condition, verbose=verbose, use_async=use_async, 
                plot_fit=plot_fit, plot_parents=plot_parents, demonstrate_parent_plotting=demonstrate_parent_plotting, constant_on_failure=constant_on_failure

            )
            for i in range(len(best_expressions)):
                self.logger.info(f"Best expression: {best_expressions[i]}, with n_chi2 {best_n_chi_squareds[i]}")
                self.logger.info(f"Initially: {results_all_dicts[i]['raw_expression']}")
         
            return {
                'trained_model': self.raw_model,
                'pruned_model': self.model,
                'train_loss': self.training_history['train_loss'],
                'results_all_dicts': results_all_dicts,
                'dataset': dataset,
                'best_expressions': best_expressions,
                'best_n_chi_squareds': best_n_chi_squareds,
                'all_results_sorted': all_results_sorted
            }
        except Exception as e:
            # Return partial results based on what was completed
            results = {}
            if hasattr(self, 'raw_model'):
                results['trained_model'] = self.raw_model
            if hasattr(self, 'model') and self.model is not None:
                results['pruned_model'] = self.model
            if hasattr(self, 'training_history') and self.training_history is not None:
                results['train_loss'] = self.training_history['train_loss']
            if 'dataset' in locals():
                results['dataset'] = dataset
            if 'best_expressions' in locals():
                results['best_expressions'] = best_expressions
                results['best_n_chi_squareds'] = best_n_chi_squareds
            if 'all_results_sorted' in locals():
                results['all_results_sorted'] = all_results_sorted
                
            self.logger.error(f"Error in pipeline: {e}, returning partial results: {list(results.keys())}")
            return results
    
    # Helper methods
    def _subst_params(self, expr_param, param_values, round_to=4):
        """
        Substitute numeric values for parameters in expressions.
        
        Args:
            expr_param: Expression string with parameter placeholders (params[i])
            param_values: List of parameter values to substitute
            
        Returns:
            String with parameters substituted with their numeric values (floats)
        """
        expr_float = expr_param
        for i in range(len(param_values)):
            expr_float = expr_float.replace(f'params[{i}]', f'{param_values[i]:.{round_to}f}')
        return expr_float

    def _prune_small_params(self, params, threshold=1e-6):
       """
       Prune parameters smaller than threshold to zero.
       
       Args:
           params: List of parameter values
           threshold: Minimum absolute value to keep (smaller values become zero)
           
       Returns:
           List of parameters with small values set to zero
       """
       return [p if abs(p) > threshold else 0 for p in params]
    
    def _convert_sympy_to_numpy(self, expr):
        """
        Convert a sympy expression to a numpy-compatible string.
        
        This function transforms a symbolic expression into a string that can be 
        evaluated using numpy functions. It handles special cases like square() functions,
        ensures all mathematical functions have the proper np. prefix, and removes 
        lambda expressions.
        
        Args:
            expr: SymPy expression or string representation of a mathematical expression
            
        Returns:
            String representation of the expression compatible with numpy evaluation
        """
        # Replace any square() functions with np.square
        if 'square(' in str(expr):
            # Replace square() with x**2 pattern
            expr = re.sub(r'square\(([^)]+)\)', r'(\1)**2', str(expr))
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
        
        # Replace any inf values with 1e8
        expr_str = expr_str.replace('inf', ' (1e8) ')
        
        return re.sub(r'lambda[^:]*:', '', expr_str)
    
    def _simplify_expression(self, formula, N=10, timeout=30):
        """
        Simplify a mathematical expression using sympy.
        
        This function attempts to algebraically simplify an expression using SymPy.
        It handles function name mapping between numpy and sympy, deals with 
        safe evaluation, and provides timeout functionality to prevent 
        simplification from taking too long.
        
        Args:
            formula: String representation of a mathematical expression
            N: Number of variables to support (creates symbols x0 to xN)
            timeout: Maximum time in seconds to attempt simplification
            
        Returns:
            String representation of the simplified expression
        """
        if formula is None or not formula.strip():
            return ""
            
        original_formula = formula
        last_good_formula = formula

        variables = symbols(f'x0:{N+1}')
        used_functions = {name: self.numpy_to_sympy[name] for name in self.numpy_to_sympy if f'{name}' in formula}
        safe_dict = {f'x{i}': variables[i] for i in range(N+1)}

        # Define symbolic variables and functions
        # if N > 1:
        #     variables = symbols(f'x0:{N+1}')
        #     safe_dict = {f'x{i}': variables[i] for i in range(N+1)}
        # else:
        #     variables = symbols(f'x')
        #     safe_dict = {f'x': variables}

        # used_functions = {name: self.numpy_to_sympy[name] for name in self.numpy_to_sympy if f'{name}' in formula}


        safe_dict.update(used_functions)  # Add only used symbolic functions
        safe_dict.update({'sp': sp})
        
        try:
            formula = formula.replace("np.", "")  # Remove "np." prefix for SymPy functions
            for key, value in self.numpy_to_sympy.items():
                # Replace function names with their sympy equivalents, but avoid replacing if already prefixed with sp.
                formula = re.sub(r'(?<!sp\.)\b' + key + r'\b(?=\()', "sp." + value.__name__, formula, flags=re.IGNORECASE)
            # Find all unknown functions in the formula without 'sp.' prefix
            unknown_functions = re.findall(r'(?<!sp\.)\b\w+\b(?=\()', formula)
            for func in unknown_functions:
                if func not in safe_dict:
                    safe_dict[func] = sp.Function(func)
                    
            self.logger.info(f"Simplifying with timeout of {timeout} seconds: {formula}")
            formula = formula.replace('inf', ' (1e8) ')
            last_good_formula = formula
            
            with stopit.ThreadingTimeout(timeout) as tt:
                try:
                    expr = simplify(eval(formula, safe_dict))
                except (SyntaxError, NameError, TypeError) as e:
                    self.logger.warning(f"Error in formula syntax: {e}")
                    return str(formula)  # Return the original formula if it can't be evaluated
                
            if tt.state == tt.TIMED_OUT:
                self.logger.warning(f"Simplification timed out after {timeout} seconds, returning unsimplified expression")
                try:
                    return str(eval(last_good_formula, safe_dict))
                except (SyntaxError, NameError, TypeError) as e:
                    self.logger.warning(f"Error in formula syntax during timeout recovery: {e}")
                    return str(last_good_formula)
                
        except Exception as e:
            self.logger.warning(f"Error simplifying expression: {e}, returning last good formula: {last_good_formula}")
            try:
                return str(eval(last_good_formula, safe_dict))
            except (SyntaxError, NameError, TypeError) as e:
                self.logger.warning(f"Error in formula syntax during exception recovery: {e}")
                return str(last_good_formula)
        
        result = str(expr)  # removes the sp. prefix
        return result
    
    def _replace_floats_with_params(self, expr_str):
        """
        Replace floating-point numbers in an expression string with symbolic parameters.
        
        This function identifies all floating point constants in an expression and
        replaces them with parameter placeholders (params[i]) to make the expression
        suitable for curve fitting.
        
        Args:
            expr_str: String representation of a mathematical expression with floating-point constants
            
        Returns:
            Tuple of (parameterized_expr_str, param_values) where:
                - parameterized_expr_str: Expression with floats replaced by params[i]
                - param_values: List of the original floating point values
        """
        # Extract float numbers using regex
        float_pattern = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?')
        float_matches = list(set(float_pattern.findall(expr_str)))
        
        # Sort floats by length to avoid partial replacements
        float_matches.sort(key=len, reverse=True)
        
        # Create symbolic parameters
        params = sp.symbols(f'param:{len(float_matches)}')
        param_values = [float(num) for num in float_matches]
        
        # Replace numeric values in the expression
        expr_str_with_params = expr_str
        for i, num in enumerate(float_matches):
            expr_str_with_params = expr_str_with_params.replace(num, f'params[{i}]')
        
        self.logger.debug(f"Converted expression with floats to expression with params: {expr_str_with_params}")
        return expr_str_with_params, param_values
    
    def _call_model_simplify(self, ranges, expr, client=None, gpt_model="openai/gpt-4o", 
                            system_prompt=None, sympy=True, numpy=False, num_answers=3, number_of_prompts=3):
        """
        Call LLM to simplify a mathematical expression within specified ranges.
        
        This function sends a mathematical expression to a language model (like GPT-4)
        and asks it to simplify the expression in multiple ways. It handles prompt
        construction, response parsing, and extracting the simplified expressions
        from potentially varied response formats.
        
        Args:
            ranges: Tuple of (min, max) values for the interval where the expression will be used
            expr: The expression to simplify
            client: LLM API client to use for the request
            gpt_model: The GPT model to use (e.g., "openai/gpt-4o")
            system_prompt: Custom system prompt to use (if None, uses a default prompt)
            sympy: Whether to request SymPy-compatible expressions (default: True)
            numpy: Whether to request NumPy-compatible expressions (default: False)
            num_answers: Number of different simplified expressions to request
            
        Returns:
            List of simplified expression strings from the LLM response
        """
        if sympy and numpy:
            raise ValueError("Cannot specify both sympy and numpy as True.")
            
        if system_prompt is None or system_prompt == "default":
            system_prompt = (
                "You are a mathematical simplification expert. Simplify the given function over the specified interval."
                "\n\nConsider these simplification strategies:"
                "\n- Taylor expansion of terms that are small in this interval"
                "\n- Remove negligible terms"
                "\n- Recognizing polynomial patterns as Taylor series terms"
                "\n- Combining similar terms and factor them when possible"
                f"\n\nYour response must provide {num_answers} different simplified version(s) following this format exactly:"
                "\n```simplified_expression_1\n[your first simplified expression here]\n```"
                "\n```simplified_expression_2\n[your second simplified expression here]\n```"
                "\n(and so on for each requested version)"
                "\nOnly include the simplified expressions inside the delimiters, nothing else. Do not include the square brackets used in the format specification, any other text, or placeholders."
                f"\nEach simplified expression should be a valid mathematical expression in {'sympy' if sympy else 'numpy'} that can be evaluated."
            )
            self.logger.info("Using default system prompt for LLM simplification")
        else:
            self.logger.info("Using provided system prompt for LLM simplification")
        
        prompt = (
            f"Please simplify this mathematical expression in {num_answers} different ways:\n\n"
            f"Function: {expr}\n"
            f"Valid interval: {ranges}\n\n"
            f"Return {num_answers} different simplified version(s) enclosed in the specified format."
        )
        
        async def get_simplifications():
            # Use provided client or instance client
            client_to_use = client if client is not None else self.client
            if client_to_use is None:
                raise ValueError("Client must be provided either during initialization or to this method call")
                
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                response = await loop.run_in_executor(
                    executor,
                    lambda: client_to_use.chat.completions.create(
                        model=gpt_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=20000,
                    )
                )
                
                out = response.choices[0].message.content
                
                # Extract expressions for all requested versions
                results = []
                for i in range(1, num_answers + 1):
                    # Try the requested format first
                    pattern = f"```simplified_expression_{i}\n(.*?)\n```"
                    match = re.search(pattern, out, re.DOTALL)
                    
                    if match:
                        results.append(match.group(1).strip())
                    else:
                        # Try alternative patterns
                        alt_patterns = [
                            f"```[ ]*(?:simplified_expression_?)?{i}?[ ]*\n(.*?)\n```",  # Various formats with numbering
                            f"```(.*?)```",  # Just find the next code block if specific numbering fails
                            f"simplified_expression_{i}:[ ]*(.*)",  # Label with number
                            f"simplified expression {i}:[ ]*(.*)",  # Alternative label format
                            f"expression {i}:[ ]*(.*)",  # Shorter label
                            f"simplified {i}:[ ]*(.*)"   # Even shorter label
                        ]
                        
                        for pattern in alt_patterns:
                            alt_match = re.search(pattern, out, re.DOTALL)
                            if alt_match:
                                # Remove any remaining backticks
                                result = alt_match.group(1).strip().replace('`', '')
                                results.append(result)
                                # Update the 'out' to remove the matched part for subsequent searches
                                out = out.replace(alt_match.group(0), "", 1)
                                break
                
                # If we couldn't find enough expressions, just return what we have
                if not results:
                    results = [out.strip().replace('`', '')]
                return results
        
        # Create and run multiple async tasks
        async def run_multiple_simplifications():
            tasks = [get_simplifications() for _ in range(number_of_prompts)]
            results_list = await asyncio.gather(*tasks)
            # Flatten the list of lists
            return [result for sublist in results_list for result in sublist]
            
        return asyncio.run(run_multiple_simplifications())

    def _fit_params(self, x_data, y_data, curve_np, params_initial, curve_ansatz_np=None, log_methods=False, log_everything=False, try_harder_jax = False, timeout_regular = 30):
        """
        Fit parameters for a given curve to data.
        
        Args:
            x_data: Input data array
            y_data: Target data array
            curve_np: Function to fit
            params_initial: Initial parameters
            curve_ansatz_np: String representation of the curve function for NumPy
            timeout_regular: Timeout for regular (not jax) fitting in seconds
            
        Returns:
            Tuple of (optimised_parameters, n_chi_squared) where:
                - optimised_parameters: Array of best-fit parameter values
                - n_chi_squared: n_chi-squared goodness of fit metric for the optimised parameters
        """
        try:
            self.logger.debug(f"Fitting parameters with initial parameters {str(params_initial[:3])[:-1]}...")
            params_opt, n_chi_squared = fit_curve_with_guess_jax(
                x_data, y_data, curve_np, params_initial,
                log_everything=log_everything,  # We don't want all logs by default
                log_methods=log_methods,      # But we do want to see which methods were used
                then_do_reg_fitting=True, 
                numpy_curve_str=curve_ansatz_np,
                timeout_regular=timeout_regular
            )
            if not np.isinf(n_chi_squared):
                return params_opt, n_chi_squared
            else:
                raise RuntimeError("Trying with multiple random initializations..." if try_harder_jax else "Trying with random initializations...")

        except RuntimeError as e:
            self.logger.debug(f"Fitting parameters in _fit_params failed: {e}")
            # Try with random initial parameters
            if try_harder_jax:
                self.logger.info("Trying harder with multiple random initializations as we got an inf n_chi_squared (4 attempts)...")
                best_params = params_initial
                best_n_chi_squared = np.inf
                num_attempts = 4 # Number of random initializations to try
                
                for i in tqdm.tqdm(range(num_attempts), desc=f"Finding best fit (current χ²: {best_n_chi_squared:.2e})"):
                    try:
                        # Use jax random if available, otherwise numpy
                        random_params = np.random.uniform(-1.0, 1.0, len(params_initial))
                        params, n_chi_squared = fit_curve_with_guess_jax(
                            x_data, y_data, curve_np, random_params,
                            log_everything=log_everything,
                            log_methods=log_methods,
                            then_do_reg_fitting=True,
                            numpy_curve_str=curve_ansatz_np,
                        )
                        
                        if n_chi_squared < best_n_chi_squared:
                            best_params = params
                            best_n_chi_squared = n_chi_squared
                            # Update tqdm description with new best chi-squared
                            i.set_description(f"Finding best fit (current χ²: {best_n_chi_squared:.2e})")
                            #if best_n_chi_squared < 1e-6:  # Early stopping if we get an excellent fit
                            break
                    except Exception as e_inner:
                        continue
                
                self.logger.info(f"Best fit after multiple attempts: n_chi_squared = {best_n_chi_squared}")
                return best_params, best_n_chi_squared
            else:
                # Just try once with random parameters
                try:
                    random_params = np.random.uniform(-1.0, 1.0, len(params_initial))
                    return fit_curve_with_guess_jax(
                        x_data, y_data, curve_np, random_params,
                        log_everything=log_everything,  # We don't want all logs by default 
                        log_methods=log_methods,      # But we do want to see which methods were used
                        then_do_reg_fitting=True, 
                        numpy_curve_str=curve_ansatz_np,
                        timeout_regular=timeout_regular
                    )
                except RuntimeError as e2:
                    self.logger.info(f"Fitting parameters in _fit_params with random initial parameters failed: {e2} after {e}")
                    return params_initial, np.inf


def run_complete_pipeline(client, f, ranges=(-np.pi, np.pi), width=[1,4,1], grid=7, k=3, 
                         train_steps=50, generations=3, gpt_model="openai/gpt-4o", device='cpu',
                         node_th=0.2, edge_th=0.2, custom_system_prompt_for_second_simplification=None, 
                         optimiser="LBFGS", population=10, temperature=0.1,
                         exit_condition=None, verbose=0, use_async=True, plot_fit=True, 
                         plot_parents=False, demonstrate_parent_plotting=False, seed=17,
                         constant_on_failure=False, simplification_gpt_model=None):
    """
    Run the complete KAN symbolic regression pipeline on a univariate function.
    
    This is a convenience function that creates a KAN_LEx instance and runs the complete pipeline.
    
    Args:
        client: Client for LLM API calls
        f: Target function to approximate
        ranges: Tuple of (min_x, max_x) for the input range
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
        optimiser: Optimiser to use for training
        population: Population size for genetic algorithm
        temperature: Temperature parameter for the genetic algorithm
        exit_condition: Exit condition for the genetic algorithm
        verbose: Verbosity level
        use_async: Whether to use async execution
        plot_fit: Whether to plot fitting results
        plot_parents: Whether to plot parent solutions
        demonstrate_parent_plotting: Whether to demonstrate parent plotting
        seed: Random seed for reproducibility
        constant_on_failure: Whether to return a constant function on failure
        simplification_gpt_model: Model to use for simplification
    Returns:
        Dictionary with complete results including models and expressions
    """
    # Validate required parameters
    if client is None:
        raise ValueError("Client must be provided")
    if f is None:
        raise ValueError("Target function f must be provided")
    
    kanlex = KANLEX(client=client, width=width, grid=grid, k=k, seed=seed, device=device)
    return kanlex.run_complete_pipeline(
        client=client, f=f, ranges=ranges,
        train_steps=train_steps, generations=generations, gpt_model=gpt_model,
        node_th=node_th, edge_th=edge_th, 
        custom_system_prompt_for_second_simplification=custom_system_prompt_for_second_simplification,
        optimiser=optimiser, population=population, temperature=temperature,
        exit_condition=exit_condition, verbose=verbose, use_async=use_async,
        plot_fit=plot_fit, plot_parents=plot_parents, 
        demonstrate_parent_plotting=demonstrate_parent_plotting,
        constant_on_failure=constant_on_failure, simplification_gpt_model=simplification_gpt_model
    )
