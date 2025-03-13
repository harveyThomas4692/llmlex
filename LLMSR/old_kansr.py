import LLMSR
import torch
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, simplify
from sympy import exp, sin, cos, log, sqrt, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, floor, ceiling, Abs
from scipy.optimize import curve_fit
import re
import logging
import io


class KANSR:
    def __init__(self, kan_model, llm_client, llm_model="openai/gpt-4o", log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.log_level = log_level
        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
        # Move http request stuff to debug
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        if kan_model.width_in[0] != 1:
            err = "The code only supports univariate functions at the moment."
            self.logger.error(err)
            raise Exception(err)

        self.raw_model = kan_model
        self.dataset = None
        self.model = None
        self.training_history = None
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.device = kan_model.device
        self.numpy_to_sympy = {
            "exp": exp, "sin": sin, "cos": cos, "log": log, "sqrt": sqrt,
            "tan": tan, "asin": asin, "acos": acos, "atan": atan,
            "sinh": sinh, "cosh": cosh, "tanh": tanh, "asinh": asinh, "acosh": acosh, "atanh": atanh,
            "floor": floor, "ceil": ceiling, "abs": Abs
        }
        self.sympy_to_numpy = {
            "exp": "np.exp", "sin": "np.sin", "cos": "np.cos", "log": "np.log", "sqrt": "np.sqrt",
            "tan": "np.tan", "asin": "np.arcsin", "acos": "np.arccos", "atan": "np.arctan",
            "sinh": "np.sinh", "cosh": "np.cosh", "tanh": "np.tanh", "asinh": "np.arcsinh", "acosh": "np.arccosh", "atanh": "np.arctanh",
            "floor": "np.floor", "ceil": "np.ceil", "abs": "np.abs"
        }
        
    
    def train_kan(self, dataset, opt="LBFGS", steps=50, prune_model=True, node_th=.2, edge_th=.2):
        self.dataset = dataset
        self.training_history = self.raw_model.fit(dataset, opt=opt, steps=steps)
        if prune_model:
            self.model = self.raw_model.prune(node_th=node_th, edge_th=edge_th)
        else:
            self.model = self.raw_model

    
    def kan_to_symbolic(self, population=10, generations=3, temperature=0.1, exit_condition=1e-3):
        """
        Converts a given kan model symbolic representations using llmsr.
        Parameters:
            population (int, optional): The population size for the genetic algorithm. Default is 10.
            generations (int, optional): The number of generations for the genetic algorithm. Default is 3.
            temperature (float, optional): The temperature parameter for the genetic algorithm. Default is 0.1.
            exit_condition (float, optional): The exit condition for the genetic algorithm. Default is 1e-3.
        Returns:
            - res_fcts (dict): A dictionary mapping layer, input, and output indices to their corresponding symbolic functions.
        """
        res, res_fcts = 'Sin', {}
        layer_connections = {0: {i: [] for i in range(self.model.width_in[0])}}
        for l in range(len(self.model.width_in) - 1):
            layer_connections[l] = {i: list(range(self.model.width_out[l-1])) if l > 0 else []  for i in range(self.model.width_in[l])}
        
        for l in range(len(self.model.width_in) - 1):
            for i in range(self.model.width_in[l]):
                for j in range(self.model.width_out[l + 1]):
                    if (self.model.symbolic_fun[l].mask[j, i] > 0. and self.model.act_fun[l].mask[i][j] == 0.):
                        raise Exception(f'({l},{i},{j}) is already symbolic. We only support non-symbolic KANS at the moment.')
                    elif (self.model.symbolic_fun[l].mask[j, i] == 0. and self.model.act_fun[l].mask[i][j] == 0.):
                        self.model.fix_symbolic(l, i, j, '0', verbose=verbose > 1, log_history=False)
                        self.logger.info(f'fixing ({l},{i},{j}) with 0')
                        symb_formula = [s.replace(f'f_{{{l},{i},{j}}}', '0') for s in symb_formula]
                        res_fcts[(l, i, j)] = None
                    else:
                        x_min, x_max, y_min, y_max = self.model.get_range(l, i, j, verbose=False)
                        x, y = self.model.acts[l][:, i].cpu().detach().numpy(), self.model.spline_postacts[l][:, j, i].cpu().detach().numpy()
                        ordered_in = np.argsort(x)
                        x, y = x[ordered_in], y[ordered_in]
                        fig, ax = plt.subplots()
                        plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                        plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                        base64_image = LLMSR.images.generate_base64_image(fig, ax, x, y)
                        self.logger.info(f"Processing KAN connection {(l,i,j)}")
                        # suppress plot by writing to random buffer
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format='png')  # Save figure to buffer instead of showing it
                        plt.close(fig)
                        mask = self.model.act_fun[l].mask
                        try:
                            res = LLMSR.run_genetic(self.llm_client, base64_image, x, y, population, generations, temperature=temperature, model=self.llm_model, system_prompt=None, elite=False, exit_condition=exit_condition, for_kan=True)
                            res_fcts[(l,i,j)] = res
                        except Exception as e:
                            self.logger.error(e)
                            res_fcts[(l,i,j)] = res
        return res_fcts
    
        
    def get_symbolic(self, population=10, generations=3, temperature=0.1, exit_condition=1e-3):
        if self.model is None:
            raise Exception("You need to train the model before fitting a symbolic expression")
        # 1.) build disctionary with all expressions ordered by score
        sym_expr = self.kan_to_symbolic(population=population, generations=generations, temperature=temperature, exit_condition=min(self.training_history['train_loss']).item())
        symb_expr_sorted = {}
        for kan_conn, sub_res in sym_expr.items():
            if sub_res is None:
                self.logger.warning(f"Could not fit a function for connection {kan_conn}")
                continue
            ordered_elements = sorted([item for sublist in sub_res for item in sublist], key=lambda item: -item['score'])
            symb_expr_sorted[kan_conn] = ordered_elements
            self.logger.info(f"Approximation for {kan_conn}: {ordered_elements[0]['ansatz'].strip()}")
            self.logger.info(f"Parameters are {np.round(ordered_elements[0]['params'], 1)}")
    
        # 2.) build dictionary for edge functions (to combine all KAN functions into one expression in the next step)
        edge_dict_top_three, edge_dict = {}, {}
        for kan_conn, sub_res in symb_expr_sorted.items():
            self.logger.info(f"KAN Connection: {kan_conn}")
            scores, ansatze, params = [pop['score'] for pop in sub_res], [pop['ansatz'].replace('*x', ' * x').replace('(x)', '(1. * x)') for pop in sub_res], [pop['params'] for pop in sub_res]
            top_three_ansatze, top_three_params, top_three_scores = [], [], []
            for s, a, p in zip(scores, ansatze, params):
                a = a.strip()
                # weed out lambda functions
                if "lambda" in a:
                    continue
                if a not in top_three_ansatze and round(s, 5) not in top_three_scores:
                    top_three_ansatze.append(a)
                    top_three_params.append(p)
                    top_three_scores.append(round(s, 5))                
                if len(top_three_ansatze) >= 3:
                    break
            top_three = []
            for a, p in zip(top_three_ansatze, top_three_params):
                top_three.append(self.subst_params(a, p))
            edge_dict_top_three[kan_conn] = top_three
            edge_dict[kan_conn] = top_three[0]
            self.logger.info([(s,t) for s,t in zip(top_three, top_three_scores)])
    
        # define an expression tree consisting of L layers, with N nodes, where node (l,n) = \sum_c expr(l,c,n)
        # the KAN convention for the triples (l,c,n) is: (layer, coming from node in l-1, going to node in layer l)
        node_tree = {}
        for l in range(len(self.model.width_in) - 1):
                for n in range(self.model.width_in[l+1]):
                    node_tree[(l, n)] = " + ".join([edge_dict[(l,c,n)].replace(' x', f' x[{l-1},{c}]' if l > 0 else f' x{c}') for c in range(self.model.width_out[l])])
        for k, v in node_tree.items():
            node_tree[k] = v.replace('+ -','- ')
        
        # 3.) Build the full KAN expression and simplify it using sympy
        full_expression = []
        for o in range(self.model.width_in[-1]):
            res = node_tree[len(self.model.width_in)-2, o]
            # traverse down lower layers
            for l in list(range(len(self.model.width_in)-2))[::-1]:
                for n in range(len([x for x in node_tree.keys() if x[0]==l])):
                    res = res.replace(f'x[{l},{n}]', f'({node_tree[l, n]})')
            full_expression.append(self.simplify_expression(res, self.model.width_in[0] - 1))
        
        # 4.) After simplification, prune (round coefficients to 4 digits), simplify again
        simplified_expressions, refitted_expressions, final_expressions = [], [], []
        for i, expr in enumerate(full_expression):
            self.logger.info("###################################################")
            self.logger.info(f"Simplifying output {i}")
            self.logger.info("KAN expression (raw):")
            self.logger.info(expr)
            f_fitted = lambda x0: eval(expr)
            # xs = np.arange(-2, 2, .1)
            # plt.plot(xs, [f(x) for x in xs], label="Original")
            # plt.plot(xs, [f_fitted(x) for x in xs], label="KANSR (raw)")
            # plt.legend();
            
            # prune and simplify
            expr = self.simplify_expression(self.subst_params(*self.replace_floats_with_params(expr)), self.model.width_in[0] - 1)
            self.logger.info("KAN expression (simplified):")
            self.logger.info(expr)
            simplified_expressions.append(expr)
            
            # refit parameters
            curve_ansatz_str, params_initial = self.replace_floats_with_params(expr)
            for k, v in self.sympy_to_numpy.items():
                curve_ansatz_str = curve_ansatz_str.replace(k, v)
            
            curve_ansatz = "lambda x0, *params: " + curve_ansatz_str
            curve = eval(curve_ansatz)
            try:
                params_opt, n_chi_squared = self.fit_curve(dataset['train_input'].numpy().flatten(), dataset['train_label'].numpy().flatten(), curve, params_initial)
                self.logger.info(f"Refitting after simplifying gave a chi^2 of {n_chi_squared:.4e}")
            except RuntimeError as e:
                self.logger.error(e)
                self.logger.info("Proceeding with unoptimised parameters")
                params_opt = params_initial
        
            # prune and simplify refitted model
            expr = self.simplify_expression(self.subst_params(curve_ansatz_str, params_opt), self.model.width_in[0] - 1)
            self.logger.info("KAN expression (refitted):")
            self.logger.info(expr)
            refitted_expressions.append(expr)
            
            # plot comparison
            # f_fitted = lambda x0: eval(expr)
            # plt.plot(xs, [f_fitted(x) for x in xs], label="KANSR (refitted)")
            
            # Ask LLM to further simplify the result and refit.
            try:
                simp_call_reply = self.call_model_simplify(prompt=f"The interval is {ranges}. The function is {expr}.")
                expr = simp_call_reply.choices[0].message.content
                curve_ansatz_str, params_initial = self.replace_floats_with_params(expr)
                curve_ansatz = "lambda x0, *params: " + curve_ansatz_str
                curve = eval(curve_ansatz)
                
                params_opt, n_chi_squared = self.fit_curve(dataset['train_input'].numpy().flatten(), dataset['train_label'].numpy().flatten(), curve, params_initial)
                self.logger.info(f"Refitting after final global simplify gave a chi^2 of {n_chi_squared:.4e}")    
                expr = self.simplify_expression(self.subst_params(curve_ansatz_str, params_opt), self.model.width_in[0] - 1)
                final_expressions.append(expr)
                # f_fitted = lambda x0: eval(expr)
                # plt.plot(xs, [f_fitted(x) for x in xs], label="KANSR (final)")
            except Exception as e:
                self.logger.error(e)
                self.logger.info("Skipping LLM improvement.")
            
            # plt.legend()
            # plt.show()
            self.logger.info("################### Final formula: ###################")
            self.logger.info(expr)
            
            return simplified_expressions, refitted_expressions, final_expressions
    
    # some helper functions
    @staticmethod
    def subst_params(a, p):
        for i in range(len(p)):
            a = a.replace(f'params[{i}]', f'{p[i]:.4f}')
        return a
    
    def simplify_expression(self, formula, N):
        # Define symbolic variables and functions
        variables = symbols(f'x0:{N+1}')
        used_functions = {name: self.numpy_to_sympy[name] for name in self.numpy_to_sympy if f'{name}' in formula}
        safe_dict = {f'x{i}': variables[i] for i in range(N+1)}
        safe_dict.update(used_functions)  # Add only used symbolic functions
        expr = simplify(eval(formula.replace("np.", ""), safe_dict))  # Remove "np." prefix for SymPy functions
        return str(expr)

    @staticmethod
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
    
    @staticmethod
    def fit_curve(x, y, curve, params_initial):
        params_opt, _ = curve_fit(curve, x, y, p0=params_initial)
        residuals = y - curve(x, *params_opt)
        n_chi_squared = np.mean((residuals ** 2) / (np.square(curve(x, *params_opt))+1e-6))
        return params_opt, n_chi_squared 

    def call_model_simplify(self, prompt=""):
        system_prompt = ("I have a function and an interval on which it is defined. Can you simplify it? Simplifications could include Taylor expansion of terms that are small in this interval, chopping terms that are small, and recognizing polynomials as the first few terms in a Taylor expansion. Please only output the final function as a string in the same format as the one that was given.")
        
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                { "role": "system", 
                 "content": system_prompt},
                {
                    "role": "user",
                    "content": [
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
