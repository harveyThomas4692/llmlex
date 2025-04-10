import os
import sys
import unittest
import base64
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
from PIL import Image

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from llmlex.images import encode_image, generate_base64_image
from tests.test_data.generate_test_data import generate_test_data

class TestImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate test data
        cls.x = np.linspace(0, 10, 100)
        cls.y = 2 * cls.x**2 + 3 * cls.x + 5
        
        # Create a test image file
        cls.test_image_path = os.path.join(tempfile.gettempdir(), 'test_image.png')
        fig, ax = plt.subplots()
        ax.plot(cls.x, cls.y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        fig.savefig(cls.test_image_path)
        plt.close(fig)
    
    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)
    
    def test_encode_image(self):
        """Test encoding an image to base64"""
        encoded = encode_image(self.test_image_path)
        
        # Check that the result is a non-empty string
        self.assertIsInstance(encoded, str)
        self.assertTrue(len(encoded) > 0)
        
        # Check that it can be decoded as base64
        decoded = base64.b64decode(encoded)
        self.assertTrue(len(decoded) > 0)
        
        # Verify the decoded content is a valid image
        image = Image.open(BytesIO(decoded))
        self.assertTrue(image.width > 0)
        self.assertTrue(image.height > 0)
    
    def test_generate_base64_image(self):
        """Test generating a base64 image from a matplotlib figure"""
        fig, ax = plt.subplots()
        base64_img = generate_base64_image(fig, ax, self.x, self.y)
        
        # Check that the result is a non-empty string
        self.assertIsInstance(base64_img, str)
        self.assertTrue(len(base64_img) > 0)
        
        # Check that it can be decoded as base64
        decoded = base64.b64decode(base64_img)
        self.assertTrue(len(decoded) > 0)
        
        # Verify the decoded content is a valid image
        image = Image.open(BytesIO(decoded))
        self.assertTrue(image.width > 0)
        self.assertTrue(image.height > 0)
        
        plt.close(fig)
    
    def test_encode_image_error_handling(self):
        """Test that encode_image properly handles missing files"""
        with self.assertRaises(FileNotFoundError):
            encode_image('nonexistent_file.png')

if __name__ == '__main__':
    unittest.main()