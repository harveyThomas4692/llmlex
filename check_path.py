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
    import LLMSR
    print(f"\nLLMSR is installed at: {os.path.dirname(LLMSR.__file__)}")
except ImportError:
    print("\nLLMSR could not be imported")