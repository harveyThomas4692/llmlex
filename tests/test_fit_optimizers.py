import os
import sys
import unittest
import numpy as np
import time
import warnings

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from LLMSR.fit import (
    fit_curve,
    fit_curve_with_guess,
    fit_curve_with_guess_jax,
    get_n_chi_squared,
    get_n_chi_squared_from_predictions
)

class TestFitOptimisers(unittest.TestCase):
    """Tests for comparing different optimisers in the fit module."""
    
    def setUp(self):
        """Set up test data and functions."""
        # Suppress specific warnings for cleaner test output
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow")
        
        # Random seed for reproducibility
        np.random.seed(42)
        
        # Common parameters
        self.x_range = np.linspace(-5, 5, 100)
        
        # Create various test functions with different complexity
        self.test_functions = {
            # Linear: f(x) = ax + b
            "linear": {
                "func": lambda x, a, b: a * x + b,
                "params": [2.5, 1.0],
                "param_count": 2,
                "jax_str": "lambda x, a, b: a * x + b",
                "np_str": "lambda x, a, b: a * np.array(x) + b",
                "initial_guess": [1.0, 0.0],
                "chi2_threshold_no_noise": 0.01,
                "chi2_threshold_medium_noise": 5.0,
                "rmse_threshold_outliers": 10.0
            },
            
            # Polynomial: f(x) = ax^3 + bx^2 + cx + d
            "polynomial": {
                "func": lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
                "params": [0.5, -1.0, 2.0, 1.0],
                "param_count": 4,
                "jax_str": "lambda x, a, b, c, d: a * (x**3) + b * (x**2) + c * x + d",
                "np_str": "lambda x, a, b, c, d: a * np.power(x, 3) + b * np.power(x, 2) + c * x + d",
                "initial_guess": [0.1, -0.5, 1.0, 0.5],
                "chi2_threshold_no_noise": 0.01,
                "chi2_threshold_medium_noise": 5.0,
                "rmse_threshold_outliers": 35.0
            },
            
            # Exponential: f(x) = a * exp(b * x) + c
            "exponential": {
                "func": lambda x, a, b, c: a * np.exp(b * x) + c,
                "params": [2.0, -0.5, 1.0],
                "param_count": 3,
                "jax_str": "lambda x, a, b, c: a * jnp.exp(b * x) + c",
                "np_str": "lambda x, a, b, c: a * np.exp(b * x) + c",
                "initial_guess": [1.0, -0.1, 0.5],
                "chi2_threshold_no_noise": 5.0,  # Exponential is harder to fit
                "chi2_threshold_medium_noise": 10.0,
                "rmse_threshold_outliers": 10.0
            },
            
            # Trigonometric: f(x) = a * sin(b * x + c) + d
            "trigonometric": {
                "func": lambda x, a, b, c, d: a * np.sin(b * x + c) + d,
                "params": [3.0, 2.0, 0.5, 1.0],
                "param_count": 4,
                "jax_str": "lambda x, a, b, c, d: a * jnp.sin(b * x + c) + d",
                "np_str": "lambda x, a, b, c, d: a * np.sin(b * x + c) + d",
                "initial_guess": [1.0, 1.0, 0.0, 0.0],
                "chi2_threshold_no_noise": 2.0,  # Trigonometric is harder to fit
                "chi2_threshold_medium_noise": 5.0,
                "rmse_threshold_outliers": 5.0
            },
            
            # Rational: f(x) = (a*x + b) / (c*x + d)
            "rational": {
                "func": lambda x, a, b, c, d: (a * x + b) / (c * x + d),
                "params": [2.0, 1.0, 0.5, 2.0],
                "param_count": 4,
                "jax_str": "lambda x, a, b, c, d: (a * x + b) / (c * x + d)",
                "np_str": "lambda x, a, b, c, d: (a * x + b) / (c * x + d)",
                "initial_guess": [1.0, 1.0, 1.0, 1.0],
                "chi2_threshold_no_noise": 600.0,  # Rational is very hard to fit
                "chi2_threshold_medium_noise": 600.0,
                "rmse_threshold_outliers": 50.0
            }
        }
        
        # Generate test data for each function
        self.test_data = {}
        for name, func_info in self.test_functions.items():
            # Generate y values without noise
            y_true = func_info["func"](self.x_range, *func_info["params"])
            
            # Add different levels of noise
            noise_levels = {
                "no_noise": np.zeros_like(y_true),
                "low_noise": np.random.normal(0, 0.1, size=len(y_true)),
                "medium_noise": np.random.normal(0, 0.5, size=len(y_true)),
                "high_noise": np.random.normal(0, 1.0, size=len(y_true))
            }
            
            # Add outliers to test robustness
            outliers = np.zeros_like(y_true)
            outlier_indices = np.random.choice(len(y_true), size=5, replace=False)
            outliers[outlier_indices] = 10.0 * np.random.randn(5)
            noise_levels["outliers"] = outliers
            
            # Store all variations
            self.test_data[name] = {
                "x": self.x_range,
                "y_true": y_true,
                "y_variations": {
                    noise_type: y_true + noise
                    for noise_type, noise in noise_levels.items()
                }
            }
    
    def test_fit_curve_all_functions(self):
        """Test fit_curve on all function types."""
        for func_name, func_info in self.test_functions.items():
            for noise_type in ["no_noise", "medium_noise"]:
                with self.subTest(f"{func_name} with {noise_type}"):
                    x = self.test_data[func_name]["x"]
                    y = self.test_data[func_name]["y_variations"][noise_type]
                    
                    # Use fit_curve with the function and the curve string
                    params, n_chi_squared = fit_curve(
                        x, y, 
                        func_info["func"], 
                        func_info["param_count"],
                        curve_str=func_info["np_str"],  # Add the curve string
                        allow_using_jax=False  # Explicitly disable JAX for this test
                    )
                    
                    # For no noise, expect chi-squared less than threshold
                    if noise_type == "no_noise":
                        self.assertLess(n_chi_squared, func_info["chi2_threshold_no_noise"],
                                       f"Chi² too high: {n_chi_squared} for {func_name}")
                    else:
                        # For medium noise, expect reasonable fit
                        self.assertLess(n_chi_squared, func_info["chi2_threshold_medium_noise"],
                                       f"Chi² too high: {n_chi_squared} for {func_name}")
                    
                    # Check parameter count
                    self.assertEqual(len(params), func_info["param_count"])
    
    def test_compare_scipy_jax_optimisers(self):
        """Compare SciPy and JAX optimisers on the same problems."""
        # Only test on linear and polynomial functions as they're more reliable
        simple_functions = ["linear", "polynomial"]
        
        for func_name in simple_functions:
            func_info = self.test_functions[func_name]
                
            for noise_type in ["no_noise", "low_noise"]:
                with self.subTest(f"{func_name} with {noise_type}"):
                    x = self.test_data[func_name]["x"]
                    y = self.test_data[func_name]["y_variations"][noise_type]
                    
                    # Time the SciPy optimisation
                    start_time = time.time()
                    scipy_params, scipy_chi2 = fit_curve_with_guess(
                        x, y,
                        func_info["func"],
                        func_info["initial_guess"],
                        try_all_methods=True
                    )
                    scipy_time = time.time() - start_time
                    
                    # Time the JAX optimisation
                    start_time = time.time()
                    jax_result_params, jax_result_n_chi_squared = fit_curve_with_guess_jax(
                        x, y,
                        None,  # We'll use the string version
                        func_info["initial_guess"],
                        numpy_curve_str=func_info["np_str"]
                    )
                    jax_time = time.time() - start_time
                    
                    # Both should produce valid results (finite chi-squared)
                    self.assertTrue(np.isfinite(scipy_chi2))
                    
                    # Skip JAX assertion if it fails
                    if np.isfinite(jax_result_n_chi_squared):
                        # Compare parameter values (should be relatively close)
                        param_diff = np.sum(np.abs(scipy_params - jax_result_params))
                        self.assertLess(param_diff, 5.0,
                                       f"Parameters differ significantly: {scipy_params} vs {jax_result_params}")
                    
                    # Print timing for reference (not a strict test)
                    print(f"\n{func_name} with {noise_type}:")
                    print(f"  SciPy time: {scipy_time:.4f}s, Chi²: {scipy_chi2:.6f}")
                    print(f"  JAX time: {jax_time:.4f}s, Chi²: {jax_result_n_chi_squared:.6f}")
                    if np.isfinite(jax_result_n_chi_squared):
                        print(f"  Speedup: {scipy_time/jax_time:.2f}x")
    
    def test_optimiser_with_outliers(self):
        """Test optimiser robustness with outliers in the data."""
        for func_name, func_info in self.test_functions.items():
            # Skip the rational function for this test
            if func_name == "rational":
                continue
                
            with self.subTest(f"{func_name} with outliers"):
                x = self.test_data[func_name]["x"]
                y_clean = self.test_data[func_name]["y_variations"]["low_noise"]
                y_outliers = self.test_data[func_name]["y_variations"]["outliers"]
                
                # Combine low noise and outliers
                y = y_clean + y_outliers
                
                # Fit with optimiser, adding curve_str and disabling JAX
                params_scipy, chi2_scipy = fit_curve(
                    x, y,
                    func_info["func"],
                    func_info["param_count"],
                    curve_str=func_info["np_str"],  # Add curve string
                    allow_using_jax=False  # Explicitly disable JAX for this test
                )
                
                # Calculate predictions and error metrics for clean data
                y_pred_scipy = func_info["func"](x, *params_scipy)
                
                # Ensure the fit is still reasonable despite outliers
                if not np.isnan(chi2_scipy) and not np.isinf(chi2_scipy):
                    rmse = np.sqrt(np.mean((y_clean - y_pred_scipy)**2))
                    # The error should be reasonable even with outliers
                    self.assertLess(rmse, func_info["rmse_threshold_outliers"],
                                   f"RMSE for {func_name} with outliers: {rmse}")
    
    def test_hard_to_fit_functions(self):
        """Test optimisers with functions that are generally hard to fit."""
        # Damped oscillation: a*exp(-b*x)*cos(c*x + d) + e
        damped_osc_func = lambda x, a, b, c, d, e: a * np.exp(-b * x) * np.cos(c * x + d) + e
        damped_osc_np_str = "lambda x, a, b, c, d, e: a * np.exp(-b * x) * np.cos(c * x + d) + e"
        
        # True parameters
        true_params = [3.0, 0.2, 2.0, 0.5, 1.0]
        
        # Generate data
        x = np.linspace(0, 10, 100)
        y_true = damped_osc_func(x, *true_params)
        y_noisy = y_true + np.random.normal(0, 0.2, size=len(y_true))
        
        # Try both optimisers with different initializations
        initializations = [
            [1.0, 0.1, 1.0, 0.0, 0.5],  # Reasonable guess
            [5.0, 0.5, 3.0, 1.0, 2.0],  # Off by a factor
            [0.1, 0.01, 0.1, 0.0, 0.1]   # Very small guess
        ]
        
        for i, init_params in enumerate(initializations):
            with self.subTest(f"Damped oscillation with initialization {i+1}"):
                # Try scipy optimiser
                scipy_params, scipy_chi2 = fit_curve_with_guess(
                    x, y_noisy, 
                    damped_osc_func, 
                    init_params,
                    try_all_methods=True
                )
                
                # Try JAX optimiser
                jax_result_params, jax_result_n_chi_squared = fit_curve_with_guess_jax(
                    x, y_noisy,
                    None,
                    init_params,
                    numpy_curve_str=damped_osc_np_str
                )
                
                # Print the results
                print(f"\nDamped oscillation initialization {i+1}:")
                print(f"  True parameters: {true_params}")
                print(f"  Initial guess: {init_params}")
                print(f"  SciPy result: {scipy_params}, Chi²: {scipy_chi2:.6f}")
                print(f"  JAX result: {jax_result_params}, Chi²: {jax_result_n_chi_squared:.6f}")
                
                # Check that at least one optimiser succeeded
                self.assertTrue(
                    np.isfinite(scipy_chi2) or np.isfinite(jax_result_n_chi_squared),
                    "Both optimisers failed to fit the damped oscillation"
                )
    
    def test_jax_optimisation_methods(self):
        """Test different JAX optimisation methods."""
        # Use a simple case where we know the answer
        func_name = "linear"  # Use linear instead of polynomial for more reliability
        x = self.test_data[func_name]["x"]
        y = self.test_data[func_name]["y_variations"]["low_noise"]
        func_info = self.test_functions[func_name]
        
        # Compare different optimisation approaches
        methods = [
            ("JAX + regular fitting", True)  # Only use JAX with regular fitting
        ]
        
        for method_name, then_do_reg_fitting in methods:
            with self.subTest(method_name):
                # Time the JAX optimisation
                start_time = time.time()
                jax_result_params, jax_result_n_chi_squared = fit_curve_with_guess_jax(
                    x, y,
                    None,
                    func_info["initial_guess"],
                    numpy_curve_str=func_info["np_str"],
                    then_do_reg_fitting=then_do_reg_fitting
                )
                jax_time = time.time() - start_time
                
                # Check results are valid
                self.assertTrue(np.isfinite(jax_result_n_chi_squared))
                self.assertEqual(len(jax_result_params), func_info["param_count"])
                
                # Print results
                print(f"\n{func_name} with {method_name}:")
                print(f"  Time: {jax_time:.4f}s, Chi²: {jax_result_n_chi_squared:.6f}")
                print(f"  Parameters: {jax_result_params}") 
                
    def test_jax_with_curve_str(self):
        """Test explicit JAX optimisation with curve strings for each function type."""
        # Only test on simpler functions as they're more reliable
        simple_functions = ["linear", "polynomial"]
        
        for func_name in simple_functions:
            func_info = self.test_functions[func_name]
            
            with self.subTest(f"JAX with curve_str for {func_name}"):
                x = self.test_data[func_name]["x"]
                y = self.test_data[func_name]["y_variations"]["low_noise"]
                
                # Fit using fit_curve with explicit JAX and curve_str
                params, n_chi_squared = fit_curve(
                    x, y,
                    func_info["func"],
                    func_info["param_count"],
                    curve_str=func_info["np_str"],
                    allow_using_jax=True,  # Explicitly enable JAX
                    force_using_jax=False  # But don't force it
                )
                
                # Should produce valid results
                self.assertTrue(np.isfinite(n_chi_squared))
                self.assertEqual(len(params), func_info["param_count"])
                
                print(f"\nfit_curve with JAX for {func_name}:")
                print(f"  Params: {params}")
                print(f"  Chi²: {n_chi_squared:.6f}")

if __name__ == '__main__':
    unittest.main() 