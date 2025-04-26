# Large Lange Models Learning Expressions (LLM_LEx)
LLM_LEx is a Python library for symbolic regression using vision-capable Large Language Models. It finds mathematical formulae to fit your data by visualizing them as graphs and using LLMs to suggest equations.

Our custom scoring function is a robust, approximately scale-invariant "normalized chi-squared" that handles both large and small values gracefully. Note that it isn't exactly scale invariance, because we actually (do/may) not want complete scale invariance. If the mean and MAD are close to zero, the score normalises by a small epsilon instead. This is so that functions that are approximately zero can be well-modelled by a fit which is simply zero.

$$n\_\chi^2 = \frac{1}{N}\sum_{i=1}^{N}\frac{(y_i - \hat{y}_i)^2}{\max(\text{global\\_scale}, \alpha |y_i|)^2}$$

where:
- $y_i$ are the actual values
- $\hat{y}_i$ are the predicted values
- $\text{global\\_scale} = \max(\text{MAD}_y, \alpha \cdot \text{mean}(|y|), \epsilon)$ - a global scale measure that never collapses to zero
- $\text{MAD}_y$ is the median absolute deviation: $\text{median}(|y_i - \text{median}(y_i)|)$
- $\alpha$ is a small fraction (default 0.01)
- $\epsilon$ is a small constant (default 1e-4) to prevent division by zero
- The denominator $\max(\text{global\\_scale}, \alpha |y_i|)^2$ provides a local scale adjustment

This metric smoothly transitions between absolute and relative error regimes. It should remain well-behaved even when variances and values approach zero.

## Installation
Clone the GIT repository, navigate to the folder and install using pip.

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

## Basic Usage

```python
import llmlex
import openai
import numpy as np
import matplotlib.pyplot as plt
import os

# Set up API client
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY") if os.getenv("OPENROUTER_API_KEY") else "<<<<<<your_api_key>>>>>>>", 
)

# Generate data
x = np.linspace(-1, 1, 50)
y = np.sin(np.pi * x) + 0.1 * np.random.randn(50)

# Generate image of data (or use your own)
fig, ax = plt.subplots()
ax.scatter(x, y)
base64_img = llmlex.images.generate_base64_image(fig, ax, x, y)

# Run symbolic regression
result = llmlex.single_call(client, base64_img, x, y, model="openai/gpt-4o")

# View results
print(f"Best function: {result['ansatz']}")
print(f"Parameters: {result['params']}")
print(f"Score: {result['score']}")

# For more complex problems, use genetic algorithm approach
populations = llmlex.run_genetic(
    client, base64_img, x, y, 
    population_size=5, num_of_generations=3,
    model="openai/gpt-4o"
)
```

## Working with KANs (Kolmogorov-Arnold Networks)

llmlex can be used to extract interpretable symbolic expressions from trained KAN models:

```python
import torch
from kan import KAN, create_dataset

# Train a KAN model (simple example)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KAN(width=[2,1,1,1], grid=7, k=3, seed=0, device=device)

# Create dataset
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2) # should be a torch function
#Initialize a KANLEX instance for the multivariate function
multivariate_kansr = KANLEX(
    client=client,
    width=[2,5,1 1],  # 2 inputs, 5 hidden nodes, 1 output
    grid=5,
    k=3,
    seed=42
)
# Create a dataset for the multivariate function
multivariate_dataset = multivariate_kansr.create_dataset(
    f=multivariate_function,
    ranges=(-3, 3),  # Same range for both variables
    n_var=2,  # Two input variables
    train_num=10000,
    test_num=1000
)

# Convert to symbolic expressions
best_expressions, best_chi_squareds, results_dicts, results_all_dicts = multivariate_kansr.get_symbolic(
    client=client,
    population=10,
    generations=5,
    temperature=0.1,
    gpt_model="openai/gpt-4o",
    verbose=1,
    use_async=True,
    plot_fit=True,
    plot_parents=True,
    demonstrate_parent_plotting=True,
    train_steps=500
)
```

`KANLEX.generate_learned_f_function` finds the symbolic expression expressed as a python program, but does not simplify the expression. For that, use `optimise_expression` in the KANSR class, or `run_complete_pipeline` for a complete end-to-end pipeline.

For a complete end-to-end symbolic regression pipeline using KANs, use the `run_complete_pipeline` function, or see the example notebook `Examples/kanlex_example.ipynb`.

```python
# Complete KAN-SR Pipeline
# Run the complete pipeline with custom parameters
results = llmlex.KANLEX.run_complete_pipeline(
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

To run tests, use the following command, which has a nice summary:

```
python tests/run_tests.py 
``` 
For the full gory details, use the following command:

```bash
python -m unittest discover -s tests
```

For tests with the OpenRouter API (costs money, but typically less than one cent per test) - you will need to create a `.env` file with your API key or set the environment variable: `OPENROUTER_API_KEY=your_actual_api_key`
To disable the API tests, run `run_tests.py` with the `--no-api` flag.
