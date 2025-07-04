import zarr
import signal
import sys
import os

def timeout_handler(signum, frame):
    print("Operation timed out!")
    sys.exit(1)

# Set up timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10 second timeout

try:
    print("Zarr version:", zarr.__version__)
    print("Starting to open zarr file...")
    
    zarr_path = "/home/yatin/Documents/Projects/forceful_tool_use/diffusion_related/robot_tool_use_diffusion_policy/data/victor/output_data_1.zarr"
    print(f"Path exists: {os.path.exists(zarr_path)}")
    
    # Try to open with different modes and options
    print("Attempting to open...")
    dirf = zarr.open(zarr_path, mode='r')
    print("Successfully opened!")
    print(f"Type: {type(dirf)}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
finally:
    signal.alarm(0)  # Cancel timeout
