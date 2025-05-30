import os
# Set DEBUG=2 to see if TinyGrad logs anything, helps identify hangs.
os.environ['DEBUG'] = '2'
# Try the other JIT import path as well, just in case.
# from tinygrad.engine.jit import TinyJit # For v0.10.3
# from tinygrad.jit import TinyJit # Older path

print("Attempting to import Tensor from tinygrad.tensor...")
try:
    from tinygrad.tensor import Tensor
    from tinygrad.dtype import dtypes
    print("Import successful.")

    print("Creating a simple tensor...")
    a = Tensor([1, 2, 3], dtype=dtypes.float32, requires_grad=False)
    print(f"Tensor a: {a.numpy()}") # Use numpy() to force realization and see output

    print("Performing a simple operation (a + 1)...")
    b = a + 1
    print(f"Tensor b: {b.numpy()}")

    print("Test complete. If you see this, basic TinyGrad ops worked.")

except Exception as e:
    print(f"An error occurred during the TinyGrad stability test: {e}")
    import traceback
    print("Traceback:")
    traceback.print_exc()

print("End of stability test script.")
