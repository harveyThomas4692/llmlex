import base64
import io
import matplotlib.pyplot as plt

def encode_image(image_path):
    '''
    Encodes an image to a base64 string.
    Args:
        image_path (str): The file path to the image to be encoded.
    Returns:
        str: The base64 encoded string of the image.
    '''
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
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
    ax.clear()  # Clear previous plot
    ax.plot(x, y, label='data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()

    # Save to buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)  # Reset buffer position
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8") # encode img into buffer
    buffer.close()  # Close buffer
    return base64_image