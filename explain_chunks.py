import zarr
import numpy as np

# Let's examine what chunk maps are for in zarr
def explain_chunk_map():
    print("=== Understanding Zarr Chunk Maps ===\n")
    
    # Open the working zarr file
    zarr_path = "./data/victor/output.zarr.zip"
    dirf = zarr.open(zarr_path, mode='r')
    
    print("Chunk Map in the code:")
    chunk_map = {
        "robot_act" : (100, 11),
        "robot_obs" : (100, 21),
        "image"     : (10, 512, 512, 4)
    }
    
    for key, chunk_shape in chunk_map.items():
        print(f"  {key}: {chunk_shape}")
    
    print("\nActual data shapes in the zarr file:")
    for key in dirf['data'].keys():
        arr = dirf['data'][key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        if hasattr(arr, 'chunks'):
            print(f"    Current chunks: {arr.chunks}")
    
    print("\n=== What are chunks? ===")
    print("Chunks are the fundamental storage units in zarr arrays.")
    print("They determine how data is stored and accessed on disk/memory.")
    print("Benefits of proper chunking:")
    print("1. Memory efficiency - only load chunks you need")
    print("2. Performance - optimized for access patterns")
    print("3. Compression - each chunk can be compressed independently")
    print("4. Parallel processing - chunks can be processed in parallel")
    
    print("\n=== Chunk Map Analysis ===")
    print("The chunk_map in the code specifies how to rechunk data when copying:")
    print("- robot_act: (100, 11) - 100 timesteps × 11 action dimensions")
    print("- robot_obs: (100, 21) - 100 timesteps × 21 observation dimensions")
    print("- image: (10, 512, 512, 4) - 10 timesteps × 512×512 pixels × 4 channels")
    
    print("\nThis chunking strategy is optimized for:")
    print("- Sequential access patterns (100 timesteps at a time)")
    print("- Memory usage during training")
    print("- Compression efficiency")
    
    # Check robot_act specifically
    if 'robot_act' in dirf['data']:
        robot_act = dirf['data']['robot_act']
        print(f"\nrobot_act details:")
        print(f"  Shape: {robot_act.shape}")
        print(f"  Current chunks: {robot_act.chunks}")
        print(f"  Proposed chunks: {chunk_map['robot_act']}")
        
        # Show first few values
        print(f"\nFirst 5 robot_act values:")
        print(robot_act[:5])

if __name__ == "__main__":
    explain_chunk_map()
