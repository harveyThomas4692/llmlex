import os
import sys
import unittest
import base64
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from LLMSR.images import encode_image, generate_base64_image
from tests.test_data.generate_test_data import generate_test_data

class TestImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate test data if it doesn't exist
        if not os.path.exists('tests/test_data/test_data.npz'):
            cls.x, cls.y, cls.base64_image = generate_test_data()
        else:
            # Load existing test data
            test_data = np.load('tests/test_data/test_data.npz')
            cls.x, cls.y = test_data['x'], test_data['y']
            with open('tests/test_data/test_image.txt', 'r') as f:
                cls.base64_image = f.read()
    
    def test_encode_image(self):
        """Test encoding an image to base64"""
        # This test requires that test_plot.png exists
        if not os.path.exists('tests/test_data/test_plot.png'):
            generate_test_data()
        
        encoded = encode_image('tests/test_data/test_plot.png')
        # Check that the result is a non-empty string
        self.assertIsInstance(encoded, str)
        self.assertTrue(len(encoded) > 0)
        
        # Check that it can be decoded as base64
        try:
            decoded = base64.b64decode(encoded)
            self.assertTrue(len(decoded) > 0)
        except Exception as e:
            self.fail(f"Base64 decoding failed: {e}")
    
    def test_generate_base64_image(self):
        """Test generating a base64 image from a matplotlib figure"""
        fig, ax = plt.subplots()
        base64_img = generate_base64_image(fig, ax, self.x, self.y)
        
        # Check that the result is a non-empty string
        self.assertIsInstance(base64_img, str)
        self.assertTrue(len(base64_img) > 0)
        
        # Check that it can be decoded as base64
        try:
            decoded = base64.b64decode(base64_img)
            self.assertTrue(len(decoded) > 0)
        except Exception as e:
            self.fail(f"Base64 decoding failed: {e}")
        
        plt.close(fig)

if __name__ == '__main__':
    unittest.main()