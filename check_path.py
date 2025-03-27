import sys
import site
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("\nSystem Path:")
for p in sys.path:
    print(f"  - {p}")

print("\nSite Packages:")
for p in site.getsitepackages():
    print(f"  - {p}")

# Try to find LLMSR
try:
    import LLM_LEx
    print(f"\nLLM_LEx is installed at: {os.path.dirname(LLM_LEx.__file__)}")
except ImportError:
    print("\nLLM_LEx could not be imported")