import subprocess
import numpy as np
import sys
import importlib.util
import os

# Note: this script was entirely vibe-coded.

# Prevent creation of __pycache__ directories
sys.dont_write_bytecode = True

def parse_cpp_stdout(output):
    """Parses the specialized stdout format from the C++ dumpTensor."""
    lines = [l.strip() for l in output.split('\n') if l.strip()]
    if not lines: return []
    
    tensors = []
    i = 0
    while i < len(lines):
        try:
            rank = int(lines[i])
            shape = list(map(int, lines[i+1].split()))
            values = list(map(float, lines[i+2].split()))
            tensors.append(np.array(values).reshape(shape))
            i += 3
        except (ValueError, IndexError):
            i += 1
    return tensors

def run_test(cpp_executable, py_reference):
    # 1. Run C++ and capture stdout
    print(f"Running C++: {cpp_executable}...")
    cpp_proc = subprocess.run([cpp_executable], capture_output=True, text=True)
    if cpp_proc.returncode != 0:
        print(f"C++ Crashed:\n{cpp_proc.stderr}")
        return False
    cpp_tensors = parse_cpp_stdout(cpp_proc.stdout)

    # 2. Dynamically import and run the Python reference
    print(f"Running Reference: {py_reference}...")
    
    # Since test_and_compare.py is in the same directory as torch_utils.py,
    # and this script is being run as the main entry point, the unittest 
    # directory is already in sys.path.
    
    module_name = os.path.splitext(os.path.basename(py_reference))[0]
    spec = importlib.util.spec_from_file_location(module_name, py_reference)
    if spec is None:
        print(f"FAILURE: Could not load Python reference file: {py_reference}")
        return False
    
    ref_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_module)
    
    if not hasattr(ref_module, 'run'):
        print(f"FAILURE: Reference script {py_reference} does not have a run() function.")
        return False
    
    py_tensors = ref_module.run()

    # 3. Compare
    if len(cpp_tensors) != len(py_tensors):
        print(f"FAILURE: Number of tensors mismatch! C++: {len(cpp_tensors)}, Python: {len(py_tensors)}")
        return False

    for idx, (cpp_data, py_data) in enumerate(zip(cpp_tensors, py_tensors)):
        if not np.allclose(cpp_data, py_data, atol=1e-5):
            print(f"FAILURE: Numerical mismatch in tensor {idx}!")
            print(f"PyTorch:\n{py_data}")
            print(f"C++:\n{cpp_data}")
            print(f"Max Diff: {np.abs(cpp_data - py_data).max()}")
            return False
    
    print(f"SUCCESS: All {len(cpp_tensors)} tensors match!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_and_compare.py <cpp_executable> <py_reference>")
        sys.exit(1)
        
    cpp_exe = sys.argv[1]
    py_ref = sys.argv[2]
    
    success = run_test(cpp_exe, py_ref)
    sys.exit(0 if success else 1)
