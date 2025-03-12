# LLM_SR

LLM_SR is a Python library for symbolic regression using vision-capable Large Language Models. It finds mathematical formulas to fit your data by visualizing it as graphs and leveraging LLMs to suggest equations. I recommend using the uv package manager to install the package - it's so much faster than pip!

## Installation

```bash
(uv) pip install .
```

Or for development:

```bash
(uv) pip install -e .
```

## Basic Usage

```python
import LLMSR
import openai
import numpy as np
import matplotlib.pyplot as plt
import os

# Set up API client
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY") if os.getenv("OPENROUTER_API_KEY") else "your_api_key", 
)

# Generate data
x = np.linspace(-1, 1, 50)
y = np.sin(np.pi * x) + 0.1 * np.random.randn(50)

# Generate image of data (or use your own)
fig, ax = plt.subplots()
ax.scatter(x, y)
base64_img = LLMSR.images.generate_base64_image(fig, ax, x, y)

# Run symbolic regression
result = LLMSR.single_call(client, base64_img, x, y, model="openai/gpt-4o")

# View results
print(f"Best function: {result['ansatz']}")
print(f"Parameters: {result['params']}")
print(f"Score: {result['score']}")

# For more complex problems, use genetic algorithm approach
populations = LLMSR.run_genetic(
    client, base64_img, x, y, 
    population_size=5, num_of_generations=3,
    model="openai/gpt-4o"
)
```

## Working with KANs (Kolmogorov-Arnold Networks)

LLM_SR can be used to extract interpretable symbolic expressions from trained KAN models:

```python
import torch
from kan import KAN, create_dataset

# Train a KAN model (simple example)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KAN(width=[2,1,1,1], grid=7, k=3, seed=0, device=device)

# Create dataset
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2) # should be a torch function
dataset = create_dataset(f, n_var=2, train_num=1000, test_num=100, device=device)
model.fit(dataset, opt="LBFGS", steps=100)

# Convert KAN model to symbolic expressions
sym_expr = LLMSR.kan_to_symbolic(
    model, client, 
    population=10, generations=3,
    gpt_model="openai/gpt-4o", 
    exit_condition=1e-3, 
    use_async=True
)

# Generate a callable Python function from symbolic expressions
learned_f_string, total_params, best_params = LLMSR.llmSR.generate_learned_f(sym_expr)
exec(learned_f_string)  # Creates learned_f function

# Optimize parameters with curve_fit
from scipy.optimize import curve_fit
popt, _ = curve_fit(
    learned_f, 
    (dataset['train_input'].cpu().numpy()[:,0], dataset['train_input'].cpu().numpy()[:,1]), 
    dataset['train_label'].cpu().numpy().flatten(), 
    p0=best_params
)
```

`generate_learned_f` finds the optimised paramters for the symbolic expression, but does not simplify the expression. For that, use `optimize_expression` in `kan_sr.py`, or `run_complete_pipeline` for a complete end-to-end pipeline.

For a complete end-to-end symbolic regression pipeline using KANs, use the `run_complete_pipeline` function, or see the example notebook `Examples/kan_sr_example.ipynb`.

```python
# Complete KAN-SR Pipeline
# Run the complete pipeline with custom parameters
results = LLMSR.kan_sr.run_complete_pipeline(
    client, f,
    ranges=x_range,
    width=[1, 4, 1],  # Use a wider network for this more complex function
    grid=7,
    k=3,
    train_steps=500,  # More training steps
    gpt_model="openai/gpt-4o",
    node_th=0.1,      # More conservative pruning
    edge_th=0.1,
    custom_system_prompt_for_second_simplification=system_prompt_for_second_simplification,
    generations = 3,
    population=10,
    plot_parents=True,
    demonstrate_parent_plotting=True
)

```

## Running Tests

To run tests, use the following command: 
```bash
python -m unittest discover -s tests
```

For tests with the OpenRouter API (costs money, but typically less than one cent per test):

1. Create a `.env` file with your API key or set the environment variable:
   ```bash
   OPENROUTER_API_KEY=your_actual_api_key
   ```

2. Enable real API tests:
   ```bash
   export LLMSR_TEST_REAL_API=1
   ```

3. Run tests:
   ```bash
   python -m unittest discover tests
   ```

Alternatively, use the test script:
```bash
python tests/run_real_api_tests.py
```
