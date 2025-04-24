import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path to import LLMSR
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from llmlex.images import generate_base64_image

def generate_test_data():
    """Generate test data for unit tests"""
    # Generate sample data
    x = np.linspace(0, 10, 100)
    # Simple polynomial: 2x^2 + 3x + 5
    y = 2 * x**2 + 3 * x + 5
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save the data
    np.savez(os.path.join(current_dir, 'test_data.npz'), x=x, y=y)
    
    # Generate and save a test image
    fig, ax = plt.subplots()
    base64_image = generate_base64_image(fig, ax, x, y)
    
    # Save the base64 image to a file
    with open(os.path.join(current_dir, 'test_image.txt'), 'w') as f:
        f.write(base64_image)
    
    # Save the plot image for reference
    plt.savefig(os.path.join(current_dir, 'test_plot.png'))
    plt.close(fig)
    
    return x, y, base64_image

if __name__ == "__main__":
    generate_test_data()