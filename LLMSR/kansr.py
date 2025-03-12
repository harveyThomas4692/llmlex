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
from scipy.optimize import curve_fit
import re
import copy
import asyncio
import concurrent.futures
import io
import stopit
import logging
from kan import KAN, create_dataset

import LLMSR.llmSR as llmSR
from LLMSR.fit import get_n_chi_squared, fit_curve_with_guess, fit_curve_with_guess_jax, test_expression_equivalence, get_n_chi_squared_from_predictions
import LLMSR.llm as llm
import tqdm

class KANSR:
    """
    A class for performing symbolic regression using Kolmogorov-Arnold Networks (KANs).
    
    This class provides methods for training KAN models, converting them to symbolic expressions,
    simplifying and optimizing these expressions, and visualizing results.
    """
    
    def __init__(self, client = None, width=None, grid=None, k=None, seed=17, symbolic_enabled=False, 
                 device='cpu', log_level=logging.INFO, model=None):
        """
        Initialize a KANSR instance.
        
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
        self.logger = logging.getLogger("LLMSR.kansr")
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
        self.node_tree = None
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
            opt: Optimizer to use
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
        
        self.logger.info(f"Training KAN model with {opt} optimizer for {steps} steps")
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
                           num_prompts_per_attempt=10, timeout_simplify=10, custom_system_prompt_for_second_simplification=None):
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
            
        Returns:
            List of best expressions, list of best n_chi-squared values, and list of result dictionaries
            containing detailed information about all expressions and their optimizations
        """
        initial_usage = llm.check_key_usage(self.client)
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_kan() first.")
            
        # Use provided client or instance client
        client_to_use = client if client is not None else self.client
        if client_to_use is None:
            raise ValueError("Client must be provided either during initialization or to this method call")
            
        # Update instance client
        self.client = client_to_use
            
        # Use n_chi_squared from model predictions as exit condition if not specified
        if exit_condition is None:
            try:
                x_data = self.dataset['train_input'].cpu().numpy()
                y_data = self.dataset['train_label'].cpu().numpy()
                predictions = self.model(torch.tensor(x_data).float()).detach().cpu().numpy()
                exit_condition = get_n_chi_squared_from_predictions(x_data, y_data, predictions)
                self.logger.info(f"Using n_chi_squared from model predictions as exit condition: {exit_condition}")
            except Exception as e:
                self.logger.warning(f"Could not calculate n_chi_squared: {e}. Using default exit condition.")
                exit_condition = 1e-3

        self.logger.info(f"Converting KAN model to symbolic expressions (exit_condition={exit_condition})")
        res = llmSR.kan_to_symbolic(
            self.model, client_to_use, population=population, generations=generations,
            temperature=temperature, gpt_model=gpt_model, exit_condition=exit_condition,
            verbose=verbose, use_async=use_async, plot_fit=plot_fit, plot_parents=plot_parents,
            demonstrate_parent_plotting=demonstrate_parent_plotting, constant_on_failure=constant_on_failure
        )
        
        self.symbolic_expressions = self._sort_symbolic_expressions(res)
        self.node_tree = self.build_expression_tree()
        self.optimized_expressions = self.optimize_expressions(client_to_use, gpt_model, x_data=self.dataset['train_input'].cpu().numpy(), y_data=self.dataset['train_label'].cpu().numpy(),
                                                                custom_system_prompt=custom_system_prompt_for_second_simplification,
                                                                prune_small_terms=True, plot_all=True, original_f=self.f,
                                                                num_prompts_per_attempt=num_prompts_per_attempt, timeout_simplify=timeout_simplify)
        best_expressions, best_n_chi_squareds, results_all_dicts = self.optimized_expressions 
        self.results_all_dicts = results_all_dicts
        final_usage = llm.check_key_usage(self.client)
        self.logger.info(f"API key usage whilst this was running: ${final_usage - initial_usage:.2f}")

        return best_expressions, best_n_chi_squareds, results_all_dicts
    
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
                expr = self._subst_params(ansatz, candidate['params'])
                
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
        
        self.node_tree = {
            "edge_dict": edge_dict,
            "top_k_edge_dicts": top_k_edge_dicts,
            "node_tree": node_tree,
            "full_expressions": full_expressions
        }
        
        return self.node_tree
    
    def optimize_expressions(self, client=None, gpt_model="openai/gpt-4o", x_data=None, y_data=None, custom_system_prompt=None, 
                            prune_small_terms=True, plot_all=True, original_f=None,
                            num_prompts_per_attempt=3, timeout_simplify=10, number_of_attempts_to_get_valid_llm=3):
        """
        Optimize and simplify the final expressions.
 
        Args:
            gpt_model: GPT model to use for simplification
            x_data: x data points (if None, uses training data)
            y_data: y data points (if None, uses training data)
            custom_system_prompt: Custom system prompt for LLM
            prune_small_terms: Whether to prune small terms
            plot_all: Whether to plot results
            original_f: Original function for comparison (optional)
            num_prompts_per_attempt: Number of prompts per attempt
            timeout_simplify: Timeout for simplification in seconds
            
        Returns:
            Tuple of (best_expressions, best_n_chi_squareds, detailed_results)
        """
        if self.node_tree is None:
            raise ValueError("Expression tree not built yet. Call build_expression_tree() first.")
            
        if x_data is None or y_data is None:
            if self.dataset is None:
                raise ValueError("No dataset available. Provide x_data and y_data or train the model first.")
            x_data = self.dataset['train_input'].cpu().numpy()
            y_data = self.dataset['train_label'].cpu().numpy()
            
        full_expressions = self.node_tree["full_expressions"]
        
        # Handle case where full_expressions is a single expression
        if not isinstance(full_expressions, list):
            full_expressions = [full_expressions]
            
        Ninputs = x_data.shape[-1] if len(x_data.shape) > 1 else 1
        
        # Create ranges for each input dimension
        if len(x_data.shape) > 1:
            ranges = [(float(np.min(x_data[:, i])), float(np.max(x_data[:, i]))) for i in range(Ninputs)]
        else:
            ranges = [(float(np.min(x_data)), float(np.max(x_data)))]
            
        if plot_all:
            fig, ax = plt.subplots()
            
        results_all_dicts = []
        lambda_xi = "lambda " + ", ".join([f"x{i}" for i in range(Ninputs)])
        
        for i, expr in enumerate(full_expressions):
            final_KAN_expression = None
            n_chi_squared_KAN_final = None
            final_LLM_expression = None
            n_chi_squared_LLM_final = None
            best_n_chi_squared = float('inf')
            best_expression = None
            best_expression_numpy = None
            best_fit_type = None  # Track the type of the best fit
            
            self.logger.info("\n###################################################")
            self.logger.info(f"Simplifying output {i}")
            self.logger.info(f"KAN expression (raw):\n{expr}")
            
            # Test raw expression
            expr_raw_numpy = self._convert_sympy_to_numpy(expr)
            f_fitted = eval(lambda_xi + ": " + expr_raw_numpy, {"np": np})
            
            if Ninputs == 1:
                xs = np.linspace(ranges[0][0], ranges[0][1], 100)
            else:
                xs = np.arange(ranges[0][0], ranges[0][1], (ranges[0][1]-ranges[0][0])/100)
                  # Calculate n_chi-squared for the raw expression
            try:
                raw_n_chi_squared = get_n_chi_squared(x_data, y_data, f_fitted, [], explain_if_inf=True, string_for_explanation=expr_raw_numpy)
                self.logger.info(f"Raw expression n_chi-squared with original data: {raw_n_chi_squared:.4e}")
            except Exception as e:
                self.logger.warning(f"Error calculating raw n_chi-squared: {e}")
                raw_n_chi_squared = float('inf')
                best_n_chi_squared = float('inf')
            
            # Initialize best values with the raw expression
            if raw_n_chi_squared < best_n_chi_squared:
                best_n_chi_squared = raw_n_chi_squared
                best_expression = expr
                best_expression_numpy = expr_raw_numpy
                best_fit_type = "raw"  # Initial best fit is the raw expression

                
            if plot_all and original_f is not None:
                try:
                    try:
                        ax.plot(xs, [original_f(torch.tensor(x)) for x in xs], label="function we're fitting")
                    except TypeError:
                        ax.plot(xs, [original_f(x) for x in xs], label="function we're fitting")
                except Exception as e:
                    self.logger.warning(f"Original function 'f' not defined; skipping plotting actual function: {e}")
                    
                ax.plot(xs, [f_fitted(x) for x in xs], label=f"KANSR (raw) chi^2: {raw_n_chi_squared:.4e}")
                ax.legend()
            
            # Prune and simplify: replace floats with parameters then simplify
            expr_simp_float_sp = self._simplify_expression(
                self._subst_params(*self._replace_floats_with_params(expr)),
                Ninputs, timeout=timeout_simplify*3
            )   
            f_expr_float_numpy = eval(lambda_xi + ": " + self._convert_sympy_to_numpy(expr_simp_float_sp), {"np": np})
            simplified_expr_n_chi_squared = get_n_chi_squared(x_data, y_data, f_expr_float_numpy, [], explain_if_inf=True, string_for_explanation=expr_simp_float_sp)
            if not np.isclose(simplified_expr_n_chi_squared, raw_n_chi_squared, atol=1e-3):
                self.logger.error(f"Problem with simplifier, simplified expression chi^2 {simplified_expr_n_chi_squared:.4e} is not close to raw expression chi^2 {raw_n_chi_squared:.4e}")
            else:
                self.logger.info(f"KAN expression (simplified, not fitted) remains the same chi^2: {simplified_expr_n_chi_squared:.4e}: {expr_simp_float_sp}")
            
            # Check if simplified expression is equivalent to raw expression
            expr_simplified_np = self._convert_sympy_to_numpy(expr_simp_float_sp)
            is_equivalent, diff_info = test_expression_equivalence(expr_raw_numpy, expr_simplified_np, lambda_xi, xs)
            if not is_equivalent:
                if isinstance(diff_info, (int, float)):
                    self.logger.info(f"Simplified expression differs from raw expression by average relative difference: {diff_info:.4e}")
                else:
                    self.logger.info(f"Could not compare simplified expression with raw expression: {diff_info}, expr_raw_numpy: {expr_raw_numpy}, expr_simplified_np: {expr_simplified_np}")
            
            # Refit parameters
            expr_simp_float_np = self._convert_sympy_to_numpy(expr_simp_float_sp)
            curve_ansatz_str_np, params_initial = self._replace_floats_with_params(expr_simp_float_np)
            
            curve_ansatz_np = lambda_xi + ", *params: " + curve_ansatz_str_np
            curve_np = eval(curve_ansatz_np, {"np": np})
            
            try:
                self.logger.info(f"Refitting simplified expression, trying to improve chi^2 from {simplified_expr_n_chi_squared:.4e}. We will try multiple random initializations if we get 'inf'")
                params_opt, n_chi_squared_after_refitting = self._fit_params(x_data, y_data, curve_np, params_initial, curve_ansatz_np, log_methods=True, log_everything=False, try_harder_jax=True)
                self.logger.info(f"Refitting: {curve_ansatz_str_np} - chi^2 after simplification and refitting: {n_chi_squared_after_refitting:.4e}")
                
                # Track best n_chi-squared
                if n_chi_squared_after_refitting < best_n_chi_squared:
                    simplified_expr = self._simplify_expression(
                        self._subst_params(curve_ansatz_str_np, params_opt), 
                        Ninputs, timeout=timeout_simplify*3
                    )
                    simplified_expr_numpy = self._convert_sympy_to_numpy(simplified_expr)
                    best_n_chi_squared = n_chi_squared_after_refitting
                    best_expression = simplified_expr
                    best_expression_numpy = simplified_expr_numpy
                    best_fit_type = "KANsimplified"  # Best fit is from KAN simplification
            except RuntimeError as e:
                params_opt = params_initial
                n_chi_squared_after_refitting = get_n_chi_squared(x_data, y_data, curve_np, params_opt, explain_if_inf=True, string_for_explanation=curve_ansatz_np)
                self.logger.warning(f"All fits failed: {e}, n_chi-squared with unoptimized parameters: {n_chi_squared_after_refitting:.4e}")
                
            # Prune small terms
            if prune_small_terms:
                prune_amount = 1e-6 if prune_small_terms is True else prune_small_terms
                self.logger.info(f"Pruning small terms, smaller than {prune_amount}")
                params_opt = self._prune_small_params(params_opt, prune_amount)
            
            # Final simplified expression
            expr_sp_after_refitting_pruning = self._simplify_expression(self._subst_params(curve_ansatz_str_np, params_opt), Ninputs)
            expr_sp_after_refitting_pruning_numpy = self._convert_sympy_to_numpy(expr_sp_after_refitting_pruning)
            f_expr_sp_after_refitting_pruning_numpy = eval(lambda_xi + ": " + expr_sp_after_refitting_pruning_numpy, {"np": np})
            n_chi_squared_KAN_final = get_n_chi_squared(x_data, y_data, f_expr_sp_after_refitting_pruning_numpy, [], explain_if_inf=True, string_for_explanation=expr_sp_after_refitting_pruning_numpy)
            
            # Store final KAN expression results
            final_KAN_expression = expr_sp_after_refitting_pruning
            
            if not np.isclose(n_chi_squared_KAN_final, n_chi_squared_after_refitting, atol=1e-3):
                self.logger.error(f"Problem with simplifier, final KAN expression chi^2 {n_chi_squared_KAN_final:.4e} is not close to simplified expression chi^2 {n_chi_squared_after_refitting:.4e}")
            self.logger.info(f"KAN expression after simplification and refitting (chi^2: {n_chi_squared_KAN_final:.4e}): {expr_sp_after_refitting_pruning}")

            # Update best expression if KAN final is better
            if n_chi_squared_KAN_final < best_n_chi_squared:
                best_n_chi_squared = n_chi_squared_KAN_final
                best_expression = expr_sp_after_refitting_pruning
                best_expression_numpy = expr_sp_after_refitting_pruning_numpy
                best_fit_type = "KANsimplified"  # Best fit is from KAN simplification
            # Plot comparison
            if plot_all:
                f_fitted = eval(lambda_xi + ": " + expr_sp_after_refitting_pruning_numpy, {"np": np})
                ax.plot(xs, [f_fitted(x) for x in xs], label=f"KANSR (simp. and refit.) chi^2: {n_chi_squared_KAN_final:.4e}")
            
            # Ask LLM to further simplify the result and refit
            try:
                # Use provided client or instance client
                client_to_use = client if client is not None else self.client
                if client_to_use is None:
                    raise ValueError("Client must be provided either during initialization or to this method call")
                
                # Update instance client
                self.client = client_to_use
                
                best_llm_expr = None
                best_llm_expr_numpy = None
                best_llm_n_chi_squared = float('inf')
                best_original_llm_expr = None
                
                for attempt_num in range(number_of_attempts_to_get_valid_llm):
                    try:
                        self.logger.info(f"LLM simplification attempt #{attempt_num+1}")
                        expr_llm_list = self._call_model_simplify(
                            ranges, best_expression, client=client_to_use, gpt_model=gpt_model, 
                            system_prompt=custom_system_prompt, 
                            sympy=True, numpy=False, 
                            num_answers=num_prompts_per_attempt
                        )
                        self.logger.info(f"LLM improvement responses: {expr_llm_list}")
                        
                        # Try each of the LLM simplified expressions
                        for sub_attempt, expr_llm in enumerate(expr_llm_list[:num_prompts_per_attempt], 1):
                            try:
                                self.logger.info(f"Trying LLM simplified expression #{sub_attempt}: {expr_llm}")
                                expr_llm_numpy = self._convert_sympy_to_numpy(expr_llm)
                                
                                # Check if LLM simplified expression is equivalent to KAN simplified expression
                                is_equivalent, diff_info = test_expression_equivalence(expr_sp_after_refitting_pruning_numpy, expr_llm_numpy, lambda_xi, xs)
                                if not is_equivalent:
                                    if isinstance(diff_info, (int, float)):
                                        self.logger.info(f"LLM simplified expression #{sub_attempt} differs from KAN simplified expression by average relative difference: {diff_info:.4e}")
                                    else:
                                        self.logger.info(f"Could not compare LLM simplified expression #{sub_attempt} with KAN simplified expression: {diff_info}")
                                else:
                                    self.logger.info(f"LLM simplified expression #{sub_attempt} is functionally equivalent to KAN simplified expression")
                                
                                # First evaluate the n_chi-squared of the LLM simplified expression without fitting
                                try:
                                    f_llm = eval(lambda_xi + ": " + expr_llm_numpy, {"np": np})
                                    n_chi_squared_no_fit = get_n_chi_squared(x_data, y_data, f_llm, [], explain_if_inf=True, string_for_explanation=expr_llm_numpy)
                                    self.logger.info(f"LLM attempt #{sub_attempt} chi^2 without fitting: {n_chi_squared_no_fit:.4e}")
                                    
                                    # Try to fit only if needed
                                    curve_ansatz_str_np, params_initial = self._replace_floats_with_params(expr_llm_numpy)
                                    curve_ansatz_np = lambda_xi + ", *params: " + curve_ansatz_str_np
                                    curve_np = eval(curve_ansatz_np, {"np": np})
                                    
                                    try:
                                        params_opt, n_chi_squared_after_fitting = self._fit_params(x_data, y_data, curve_np, params_initial, curve_ansatz_np, log_methods=True, log_everything=False)
                                        self.logger.info(f"LLM attempt #{sub_attempt} chi^2 after fitting: {n_chi_squared_after_fitting:.4e} for {curve_ansatz_np}, params opt.: {params_opt[:3]}")
                                    except Exception as fit_error:
                                        self.logger.warning(f"Fitting failed for LLM expression #{sub_attempt}: {fit_error}")
                                        # Use the n_chi-squared without fitting if fitting fails
                                        n_chi_squared_after_llm_fitting = n_chi_squared_no_fit
                                        params_opt = params_initial
                                    
                                    if n_chi_squared_after_llm_fitting < best_llm_n_chi_squared:
                                        best_llm_n_chi_squared = n_chi_squared_after_llm_fitting
                                        params_opt = self._prune_small_params(params_opt, prune_amount)
                                        best_llm_expr = self._simplify_expression(
                                            self._subst_params(curve_ansatz_str_np, params_opt), 
                                            Ninputs, timeout=timeout_simplify
                                        )
                                        best_llm_expr_numpy = self._convert_sympy_to_numpy(best_llm_expr)
                                        best_original_llm_expr = expr_llm
                                        
                                        # If we found an excellent fit, break out early
                                        if n_chi_squared_after_llm_fitting < 1e-6:
                                            self.logger.info(f"Found excellent fit with n_chi_squared < 1e-6: {n_chi_squared_after_llm_fitting:.4e}")
                                            break
                                except Exception as e:
                                    self.logger.warning(f"Error evaluating n_chi-squared for LLM expression #{sub_attempt}: {e}")
                            except Exception as e:
                                self.logger.warning(f"Error with LLM expression #{sub_attempt}: {e}")
                        
                        # If we found a valid expression, break out of the retry loop
                        if best_llm_expr is not None:
                            break
                        
                        self.logger.info(f"Attempt #{attempt_num+1} failed to produce a valid expression. {'Retrying...' if attempt_num < 2 else 'No more retries.'}")
                        
                    except Exception as e:
                        self.logger.warning(f"Error in LLM simplification attempt #{attempt_num+1}: {e}")
                        if attempt_num < number_of_attempts_to_get_valid_llm-1:
                            self.logger.info(f"Retrying... {attempt_num+1}/{number_of_attempts_to_get_valid_llm}")
                        else:
                            self.logger.info("No more retries.")
                
                # If we found a valid LLM expression
                if best_llm_expr is not None:
                    final_LLM_expression = best_llm_expr_numpy
                    n_chi_squared_LLM_final = best_llm_n_chi_squared
                    
                    # Check if this is the best overall expression
                    if best_llm_n_chi_squared < best_n_chi_squared:
                        best_n_chi_squared = best_llm_n_chi_squared
                        best_expression = best_llm_expr
                        best_expression_numpy = best_llm_expr_numpy
                        best_fit_type = "LLMsimplified"  # Best fit is from LLM simplification
                    
                    # Plot if requested
                    if plot_all:
                        f_fitted = eval(lambda_xi + ": " + best_llm_expr_numpy, {"np": np})
                        ax.plot(xs, [f_fitted(x) for x in xs], label=f"KANSR (after LLM simp. and refit.) chi^2: {best_llm_n_chi_squared:.4e}")
                    self.logger.info(f'Final LLM response, chi^2 {best_llm_n_chi_squared:.4e} simplified and refitted: {best_llm_expr}, from model response {best_original_llm_expr}')
                else:
                    # If all attempts failed
                    self.logger.warning("All LLM simplifications failed to fit properly")
                    final_LLM_expression = None
                    n_chi_squared_LLM_final = None
            except Exception as e:
                self.logger.warning(f"Skipping LLM improvement: {e}")
                final_LLM_expression = None
                n_chi_squared_LLM_final = None
            
            if plot_all:
                ax.legend()
                plt.show()
                
            self.logger.info(f"\n###############################\n# Final formula for output {i}: #\n###############################")
            self.logger.info(f"Best expression chi^2 {best_n_chi_squared:.3e} from {best_fit_type} fit: {best_expression}")
            result_dict = {
                'raw_expression': expr_raw_numpy,
                'raw_n_chi_squared': raw_n_chi_squared,
                'final_KAN_expression': final_KAN_expression,
                'n_chi_squared_KAN_final': n_chi_squared_KAN_final,
                'final_LLM_expression': final_LLM_expression,
                'n_chi_squared_LLM_final': n_chi_squared_LLM_final,
                'best_expression': best_expression_numpy,
                'best_n_chi_squared': best_n_chi_squared,
                'best_fit_type': best_fit_type
            }
            results_all_dicts.append(result_dict)
        
        best_expressions = [result_dict['best_expression'] for result_dict in results_all_dicts]
        best_n_chi_squareds = [result_dict['best_n_chi_squared'] for result_dict in results_all_dicts]
        
        return best_expressions, best_n_chi_squareds, results_all_dicts
    
    def plot_results(self, ranges=None, result_dict=None, dataset=None, title="KAN Symbolic Regression Results", 
                    plotmaxmin=[[None, None], [None, None]]):
        """
        Plot the original function and the approximations.
        
        Args:
            ranges: Tuple of (min_x, max_x) for the input range. If None, will try to derive from dataset.
            result_dict: Dictionary with results from optimize_expressions. If None, will use self.results_all_dicts[0] if available.
            dataset: Optional dataset to use for plotting. If None, will use self.dataset if available.
            title: Plot title
            plotmaxmin: Limits for the plot [[xmin, xmax], [ymin, ymax]]
            
        Returns:
            matplotlib figure and axes objects
        """
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
                    if len(x_data.shape) > 1:
                        # Multiple input variables, use first dimension
                        ranges = (float(np.min(x_data[:, 0])), float(np.max(x_data[:, 0])))
                    else:
                        ranges = (float(np.min(x_data)), float(np.max(x_data)))
                    self.logger.info(f"Using ranges derived from dataset: {ranges}")
                except Exception as e:
                    self.logger.warning(f"Could not derive ranges from dataset: {e}")
                    
            # If still no ranges, use default
            if ranges is None:
                ranges = (-np.pi, np.pi)
                self.logger.warning(f"No ranges provided or derivable from dataset. Using default: {ranges}")
        else:
            self.logger.info(f"Using provided ranges: {ranges}")
 
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
        
        # Try to plot the simplified expression
        try:
            idx_to_plot = 0
            raw_expr = result_dict['raw_expression']
            raw_n_chi_squared = result_dict['raw_n_chi_squared']
            y_raw = eval(raw_expr, {"np": np, "x0": x})
            self.logger.info('Plotting raw expression')
            ax.plot(x, y_raw, 'orange', dashes=[4, 2], label=f'Raw expression from LLMSR (χ²={raw_n_chi_squared:.5e})', linewidth=2)

            simplified_by_KAN_expr = result_dict['final_KAN_expression'][0]
            n_chi_squared = result_dict['n_chi_squared_KAN_final'][0]
            y_simplified = eval(simplified_by_KAN_expr, {"np": np, "x0": x})
            self.logger.info('Plotting simplified by KAN expression')
            ax.plot(x, y_simplified, 'g-.', dashes=[3, 1, 1, 1], label=f'Simplified by KAN (χ²={n_chi_squared:.5e})', linewidth=2)

            simplified_by_LLM_expr = result_dict['final_LLM_expression'][0]
            n_chi_squared = result_dict['n_chi_squared_LLM_final'][0]
            y_simplified = eval(simplified_by_LLM_expr, {"np": np, "x0": x})
            self.logger.info('Plotting simplified by LLM expression')
            ax.plot(x, y_simplified, 'c--', dashes=[2, 1], label=f'Simplified by LLM and refitted (χ²={n_chi_squared:.5e})', linewidth=2)
        except Exception as e:
            self.logger.warning(f"Error plotting simplified expression: {e}")
            
        # Try to plot the model and pruned model predictions if they're part of this instance
        # (for tests, we may not have actual models)
        if y_true is not None:  # Only try to calculate chi-squared if we have true values
            try:
                if hasattr(self, 'raw_model') and self.raw_model is not None:
                    try:
                        model_preds = self.raw_model(torch.tensor(x).reshape(-1, 1).float()).detach().numpy().flatten()
                        try:
                            chi_squared = get_n_chi_squared_from_predictions(x, y_true, model_preds)
                            ax.plot(x, model_preds, 'b:', dashes=[1, 1], label=f'KAN Model {chi_squared:.5e}', linewidth=2)
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
        
        return fig, ax
    
    def run_complete_pipeline(self, client=None, f=None, ranges=(-np.pi, np.pi), train_steps=50, 
                             generations=3, gpt_model="openai/gpt-4o", node_th=0.2, edge_th=0.2, 
                             custom_system_prompt_for_second_simplification=None, optimizer="LBFGS", 
                             population=10, temperature=0.1, exit_condition=None, verbose=0, 
                             use_async=True, plot_fit=True, plot_parents=False, demonstrate_parent_plotting=False, constant_on_failure=False):
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
            optimizer: Optimizer to use for training
            population: Population size for genetic algorithm
            temperature: Temperature parameter for the genetic algorithm
            exit_condition: Exit condition for the genetic algorithm
            verbose: Verbosity level
            use_async: Whether to use async execution
            plot_fit: Whether to plot fitting results
            plot_parents: Whether to plot parent solutions
            demonstrate_parent_plotting: Whether to demonstrate parent plotting
            constant_on_failure: Whether to return a constant function on failure
            
        Returns:
            Dictionary with complete results including models and expressions
        """
        try:
            # Use provided client or fallback to the instance's client
            client_to_use = client if client is not None else self.client
            if client_to_use is None:
                raise ValueError("Client must be provided either during initialization or to this method call")
                
            # Use client_to_use for operations that need a client
            self.client = client_to_use  # Update the instance's client
                
            # 1. Create the dataset
            if f is None:
                raise ValueError("Target function f must be provided")
                
            dataset = self.create_dataset(f, ranges=ranges, n_var=1, train_num=10000, test_num=1000)
            
            # 2. Train the model
            self.train_kan(dataset, opt=optimizer, steps=train_steps, prune=True, node_th=node_th, edge_th=edge_th)
            
            self.logger.info("Trained model:")
            self.raw_model.plot()
            self.logger.info("Pruned model:")
            self.model.plot()
            
            # 3. Convert to symbolic expressions
            self.get_symbolic(
                client=client_to_use, population=population, generations=generations, 
                temperature=temperature, gpt_model=gpt_model,
                exit_condition=exit_condition, verbose=verbose, use_async=use_async, 
                plot_fit=plot_fit, plot_parents=plot_parents, demonstrate_parent_plotting=demonstrate_parent_plotting, constant_on_failure=constant_on_failure
            )
            
            # 4. Build expression tree
            self.build_expression_tree(top_k=3)
            
            # 5. Optimize expression
            # Convert training data to numpy arrays for optimization
            x_data = dataset['train_input'].cpu().numpy().flatten()
            y_data = dataset['train_label'].cpu().numpy().flatten()
            
            # Optimize and simplify the expression
            best_expressions, best_n_chi_squareds, result_dicts = self.optimize_expressions(
                client=client_to_use, gpt_model=gpt_model, x_data=x_data, y_data=y_data, 
                custom_system_prompt=custom_system_prompt_for_second_simplification,
                prune_small_terms=True, original_f=self.f, plot_all=True
            )
            
            # Print the results
            result_dict = result_dicts[0]
            self.logger.info(f"Best expression: {result_dict['best_expression']}, at index {best_index}, with chi^2 {result_dict['best_n_chi_squared']}")
            self.logger.info(f"Initially: {result_dict['raw_expression']}")
            self.logger.info(f"Refitting all coefficients in KAN: {result_dict['final_KAN_expression'][best_index]}, chi^2 {result_dict['n_chi_squared_KAN_final'][best_index]}")
            self.logger.info(f"Simplifying by LLM and refitting again: {result_dict['final_LLM_expression'][best_index]}, chi^2 {result_dict['n_chi_squared_LLM_final'][best_index]}")
            
            return {
                'trained_model': self.raw_model,
                'pruned_model': self.model,
                'train_loss': self.training_history['train_loss'],
                'symbolic_expressions': self.symbolic_expressions,
                'node_tree': self.node_tree,
                'result_dict': result_dict,
                'dataset': dataset,
                'best_expressions': best_expressions,
                'best_n_chi_squareds': best_n_chi_squareds
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
            if hasattr(self, 'symbolic_expressions') and self.symbolic_expressions is not None:
                results['symbolic_expressions'] = self.symbolic_expressions
            if hasattr(self, 'node_tree') and self.node_tree is not None:
                results['node_tree'] = self.node_tree
            if 'result_dict' in locals():
                results['result_dict'] = result_dict
            if 'dataset' in locals():
                results['dataset'] = dataset
            if 'best_expressions' in locals():
                results['best_expressions'] = best_expressions
                results['best_n_chi_squareds'] = best_n_chi_squareds
                
            self.logger.error(f"Error in pipeline: {e}, returning partial results: {list(results.keys())}")
            return results    
    # Helper methods
    def _subst_params(self, a, p):
        """
        Substitute numeric values for parameters in expressions.
        
        Args:
            a: Expression string with parameter placeholders
            p: List of parameter values to substitute
            
        Returns:
            String with parameters substituted with their numeric values
        """
        for i in range(len(p)):
            a = a.replace(f'params[{i}]', f'{p[i]:.4f}')
        return a

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
        expr_str = NumPyPrinter({'strict': False}).doprint(expr)
        
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
        # Define symbolic variables and functions
        variables = symbols(f'x0:{N+1}')
        used_functions = {name: self.numpy_to_sympy[name] for name in self.numpy_to_sympy if f'{name}' in formula}
        safe_dict = {f'x{i}': variables[i] for i in range(N+1)}
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
                    
            self.logger.info(f"Simplifying {formula} with timeout of {timeout} seconds")
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
            expr_str: String representation of a mathematical expression
            
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
        for i, num in enumerate(float_matches):
            expr_str = expr_str.replace(num, f'params[{i}]')
        
        self.logger.debug(f"Converted to numpy, and replaced the new floats with 'params': {expr_str}")
        return expr_str, param_values
    
    def _call_model_simplify(self, ranges, expr, client=None, gpt_model="openai/gpt-4o", 
                            system_prompt=None, sympy=True, numpy=False, num_answers=3):
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
                "\n- Removing negligible terms"
                "\n- Recognizing polynomial patterns as Taylor series terms"
                "\n- Combining like terms and factoring when possible"
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
                        max_tokens=4096,
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
        
        # Run the async function and get the result
        result = asyncio.run(get_simplifications())
        return result

    def _fit_params(self, x_data, y_data, curve_np, params_initial, curve_ansatz_np=None, log_methods=False, log_everything=False, try_harder_jax = False):
        """
        Fit parameters for a given curve to data.
        
        Args:
            x_data: Input data array
            y_data: Target data array
            curve_np: Function to fit
            params_initial: Initial parameters
            curve_ansatz_np: String representation of the curve function for NumPy
            
        Returns:
            Tuple of (optimized_parameters, n_chi_squared) where:
                - optimized_parameters: Array of best-fit parameter values
                - n_chi_squared: n_chi-squared goodness of fit metric for the optimized parameters
        """
        try:
            self.logger.debug(f"Fitting parameters with initial parameters {params_initial[0:3]}...")
            params_opt, n_chi_squared = fit_curve_with_guess_jax(
                x_data, y_data, curve_np, params_initial,
                log_everything=log_everything,  # We don't want all logs by default
                log_methods=log_methods,      # But we do want to see which methods were used
                then_do_reg_fitting=True, 
                numpy_curve_str=curve_ansatz_np
            )
            if not try_harder_jax or not np.isinf(n_chi_squared):
                return params_opt, n_chi_squared
            else:
                raise RuntimeError("Trying harder with multiple random initializations...")

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
                            numpy_curve_str=curve_ansatz_np
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
                        numpy_curve_str=curve_ansatz_np
                    )
                except RuntimeError as e2:
                    self.logger.info(f"Fitting parameters in _fit_params with random initial parameters failed: {e2} after {e}")
                    return params_initial, np.inf


def run_complete_pipeline(client, f, ranges=(-np.pi, np.pi), width=[1,4,1], grid=7, k=3, 
                         train_steps=50, generations=3, gpt_model="openai/gpt-4o", device='cpu',
                         node_th=0.2, edge_th=0.2, custom_system_prompt_for_second_simplification=None, 
                         optimizer="LBFGS", population=10, temperature=0.1,
                         exit_condition=None, verbose=0, use_async=True, plot_fit=True, 
                         plot_parents=False, demonstrate_parent_plotting=False, seed=17,
                         constant_on_failure=False):
    """
    Run the complete KAN symbolic regression pipeline on a univariate function.
    
    This is a convenience function that creates a KANSR instance and runs the complete pipeline.
    
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
        optimizer: Optimizer to use for training
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
        
    Returns:
        Dictionary with complete results including models and expressions
    """
    kansr = KANSR(client=client, width=width, grid=grid, k=k, seed=seed, device=device)
    return kansr.run_complete_pipeline(
        client=client, f=f, ranges=ranges,
        train_steps=train_steps, generations=generations, gpt_model=gpt_model,
        node_th=node_th, edge_th=edge_th, 
        custom_system_prompt_for_second_simplification=custom_system_prompt_for_second_simplification,
        optimizer=optimizer, population=population, temperature=temperature,
        exit_condition=exit_condition, verbose=verbose, use_async=use_async,
        plot_fit=plot_fit, plot_parents=plot_parents, 
        demonstrate_parent_plotting=demonstrate_parent_plotting,
        constant_on_failure=constant_on_failure
    )
