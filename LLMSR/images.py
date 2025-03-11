import base64
import io
import matplotlib.pyplot as plt
import logging
import os
from LLMSR.response import fun_convert
# Get module logger
logger = logging.getLogger("LLMSR.images")

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

def generate_base64_image_with_parents( x, y, parent_functions, fig = None, ax = None):
    """
    Generates a base64 encoded PNG image with data and parent functions.
    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes.Axes): The matplotlib axes object.
        x (list or numpy.ndarray): The x data for the plot.
        y (list or numpy.ndarray): The y data for the plot.
        parent_functions (list): List of functions that take x as input.
    Returns:
        str: The base64 encoded string of the PNG image.
    """
    x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)

    logger.debug(f"Generating base64 image with {len(parent_functions)} parent functions")
    
    try:
        # Clear and prepare the plot
        logger.debug("Preparing plot")
        if ax is not None:
            ax.clear()  # Clear previous plot
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
            x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
            plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
            plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
        # Plot parent functions in faded colors
        colors = ['red', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        for i, func in enumerate(parent_functions):
            func_actual = fun_convert(func[0])[0]
            y_func = func_actual(x, *func[1])
            color_idx = i % len(colors)
            ax.plot(x, y_func, color=colors[color_idx], alpha=0.3, label=f'curve_{i}', linestyle=':', dashes=(i+1, 1))
        
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
        plt.close(fig)  # Ensure the plot is not displayed
        
        logger.debug(f"Successfully generated base64 image: {len(base64_image)} characters")
        return base64_image
        
    except Exception as e:
        logger.error(f"Error generating base64 image with parents: {e}", exc_info=True)
        raise