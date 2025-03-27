import base64
import io
import matplotlib.pyplot as plt
import logging
import os
from LLM_LEx.response import fun_convert
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# Get module logger
logger = logging.getLogger("LLMLEx.images")

def encode_image(image_path):
    '''
    Encodes an image to a base64 string.
    Args:
        image_path (str): The file path to the image to be encoded.
    Returns:
        str: The base64 encoded string of the image.
    '''
    logger.debug(f"Encoding image from path: {image_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Get file size for logging
        file_size = os.path.getsize(image_path)
        logger.debug(f"Image file size: {file_size} bytes")
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_string = base64.b64encode(image_data).decode("utf-8")
            
        logger.debug(f"Successfully encoded image: {len(base64_string)} characters")
        return base64_string
        
    except Exception as e:
        logger.error(f"Error encoding image: {e}", exc_info=True)
        raise
    
def generate_base64_image(fig, ax, x, y):
    """
    Generates a base64 encoded PNG image from a matplotlib figure and axes.
    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes.Axes): The matplotlib axes object.
        x (list or numpy.ndarray): The x data for the plot.
        y (list or numpy.ndarray): The y data for the plot.
    Returns:
        str: The base64 encoded string of the PNG image.
    """
    logger.debug(f"Generating base64 image from plot data: {len(x)} points")
    
    try:
        # Clear and prepare the plot
        logger.debug("Preparing plot")
        ax.clear()  # Clear previous plot
        ax.plot(x, y, label='data')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()

        # Save figure to buffer
        logger.debug("Saving figure to buffer")
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100)
        buffer_size = buffer.tell()
        logger.debug(f"Buffer size: {buffer_size} bytes")
        
        # Reset buffer position and encode
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Clean up
        buffer.close()
        
        logger.debug(f"Successfully generated base64 image: {len(base64_image)} characters")
        return base64_image
        
    except Exception as e:
        logger.error(f"Error generating base64 image: {e}", exc_info=True)
        raise
def generate_base64_image(fig, ax, x, y):
    """
    Generates a base64 encoded PNG image from a matplotlib figure and axes.
    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes.Axes): The matplotlib axes object.
        x (list or numpy.ndarray): The x data for the plot.
        y (list or numpy.ndarray): The y data for the plot.
    Returns:
        str: The base64 encoded string of the PNG image.
    """
    logger.debug(f"Generating base64 image from plot data: {len(x)} points")
    
    try:
        # Clear and prepare the plot
        logger.debug("Preparing plot")
        ax.clear()  # Clear previous plot
        ax.plot(x, y, label='data')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()

        # Save figure to buffer
        logger.debug("Saving figure to buffer")
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100)
        buffer_size = buffer.tell()
        logger.debug(f"Buffer size: {buffer_size} bytes")
        
        # Reset buffer position and encode
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Clean up
        buffer.close()
        
        logger.debug(f"Successfully generated base64 image: {len(base64_image)} characters")
        return base64_image
        
    except Exception as e:
        logger.error(f"Error generating base64 image: {e}", exc_info=True)
        raise

def generate_base64_image_with_parents( x, y, parent_functions, fig = None, ax = None, actually_plot = False, title_override = None):
    """
    Generates a base64 encoded PNG image with data and parent functions.
    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes.Axes): The matplotlib axes object.
        x (list or numpy.ndarray): The x data for the plot.
        y (list or numpy.ndarray): The y data for the plot.
        parent_functions (list): List of functions that take x as input.
        actually_plot (bool, optional): Whether to display the plot. Defaults to False.
        title_override (str, optional): Custom title for the plot. Defaults to None.
        fig (matplotlib.figure.Figure, optional): The matplotlib figure object. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The matplotlib axes object. Defaults to None.
    Returns:
        str: The base64 encoded string of the PNG image.
    """
    logger.debug(f"Generating base64 image with {len(parent_functions)} parent functions")
    
    try:
        # Clear and prepare the plot
        logger.debug("Preparing plot")
        if ax is not None:
            ax.clear()  # Clear previous plot
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
            plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
        else:
            # Use the range from the main function
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            
        # Plot parent functions in faded colors
        colors = ['red', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        for i, func in enumerate(parent_functions):
            func_actual = fun_convert(func[0])[0]
            y_func = func_actual(x, *func[1])
            color_idx = i % len(colors)
            ax.plot(x, y_func, color=colors[color_idx], alpha=0.7, label=f'curve_{i}', linestyle=':', dashes=(i+1, 1))
        
        # Plot main data in bright blue
        ax.plot(x, y, color='blue', linewidth=2, label='data to be fitted')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()

        # Save figure to buffer
        logger.debug("Saving figure to buffer")
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100)
        buffer_size = buffer.tell()
        logger.debug(f"Buffer size: {buffer_size} bytes")
        
        # Reset buffer position and encode
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Clean up
        buffer.close()
        if actually_plot:
            if title_override is not None:
                plt.title(title_override)
            plt.show()
        else:
            plt.close(fig)  # Ensure the plot is not displayed
        
        logger.debug(f"Successfully generated base64 image: {len(base64_image)} characters")
        return base64_image
        
    except Exception as e:
        logger.error(f"Error generating base64 image with parents: {e}", exc_info=True)
        raise



def plot_3d_function(x0_range, x1_range, z_values, test_data, title, cmap='viridis', ax=None):
    """
    Plot a 3D surface with test data points.
    
    Args:
        x0_range (numpy.ndarray): Range of values for the first input dimension.
        x1_range (numpy.ndarray): Range of values for the second input dimension.
        z_values (numpy.ndarray): 2D array of function values corresponding to the meshgrid of x0_range and x1_range.
        test_data (tuple): Tuple containing (test_x0, test_x1, test_y) arrays of test data points.
        title (str): Title for the plot.
        cmap (str, optional): Colormap to use for the surface. Defaults to 'viridis'.
        ax (matplotlib.axes.Axes, optional): Existing axes to plot on. If None, creates new figure and axes.
        
    Returns:
        tuple: (surf, ax) where surf is the plotted surface and ax is the matplotlib axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for plotting
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    # Plot the function surface
    surf = ax.plot_surface(X0, X1, z_values, cmap=cmap, alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Plot the actual test data points
    test_x0, test_x1, test_y = test_data
    ax.scatter(test_x0, test_x1, test_y, c='red', marker='o', s=10, label='True data')
    
    # Add labels and title
    ax.set_xlabel('X0', fontsize=12, labelpad=10)
    ax.set_ylabel('X1', fontsize=12, labelpad=10)
    ax.set_zlabel('Y', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Function Value', fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    return surf, ax

def calculate_mse(func, params, inputs, true_outputs):
    """Calculate mean squared error between function predictions and true values."""
    predictions = np.array([func((x0, x1), *params) for x0, x1 in zip(inputs[0], inputs[1])])
    mse = np.mean((predictions - true_outputs)**2)
    return mse, predictions
