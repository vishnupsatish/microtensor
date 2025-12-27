import subprocess
import numpy as np
import sys

def parse_stdout(output):
    lines = [l.strip() for l in output.split('\n') if l.strip()]
    if not lines: return None
    
    rank = int(lines[0])
    shape = list(map(int, lines[1].split()))
    values = list(map(float, lines[2].split()))
    return np.array(values).reshape(shape)

def run_test(cpp_executable, py_reference):
    # 1. Run C++ and capture stdout
    print(f"🚀 Running C++: {cpp_executable}...")
    cpp_proc = subprocess.run([cpp_executable], capture_output=True, text=True)
    if cpp_proc.returncode != 0:
        print(f"❌ C++ Crashed:\n{cpp_proc.stderr}")
        return False
    cpp_data = parse_stdout(cpp_proc.stdout)

    # 2. Run PyTorch Reference and capture stdout
    print(f"🔥 Running Reference: {py_reference}...")
    py_proc = subprocess.run([sys.executable, py_reference], capture_output=True, text=True)
    if py_proc.returncode != 0:
        print(f"❌ Python Crashed:\n{py_proc.stderr}")
        return False
    py_data = parse_stdout(py_proc.stdout)

    # 3. Compare
    if np.allclose(cpp_data, py_data, atol=1e-5):
        print("✅ SUCCESS: Outputs match!")
        return True
    else:
        print("❌ FAILURE: Numerical mismatch!")
        print(f"PyTorch:\n{py_data}")
        print(f"C++:\n{cpp_data}")
        print(f"Max Diff: {np.abs(cpp_data - py_data).max()}")
        return False

if __name__ == "__main__":
    cpp_exe = sys.argv[1]
    py_ref = sys.argv[2]
    
    success = run_test(cpp_exe, py_ref)
    sys.exit(0 if success else 1)