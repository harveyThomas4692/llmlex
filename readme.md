# Large Lange Models Learning Expressions (LLM-LEx)
LLM-LEx is a Python library for symbolic regression using vision-capable Large Language Models. It finds mathematical formulae to fit your data by visualizing them as graphs and using LLMs to suggest equations.

See our paper for further details: https://arxiv.org/abs/2505.07956

![](logo.png)

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

Many examples can be found in the example folders

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

# initialize KAN with G=3
model = KAN(width=[2,1,1,1], grid=7, k=3, seed=0, device=device, symbolic_enabled=False)

# create dataset
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, train_num=10000, test_num=1000, device=device)
res = model.fit(dataset, opt="LBFGS", steps=100);

# run llmlex on each edge

sym_expr = llmlex.kan_to_symbolic(model, client, gpt_model="openai/gpt-4o", exit_condition=min(res['train_loss']).item(), use_async=True, population=10, generations=3, temperature=0.1)#

best_expressions, best_chi_squareds, results_dicts, results_all_dicts = multivariate_kansr.get_symbolic(
    client=client,
    population=5,
    generations=2,
    temperature=0.1,
    gpt_model="openai/gpt-4o",
    verbose=1,
    use_async=True,
    plot_fit=True,
    plot_parents=True,
    demonstrate_parent_plotting=True
)

print(best_expressions, best_chi_squareds)

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
