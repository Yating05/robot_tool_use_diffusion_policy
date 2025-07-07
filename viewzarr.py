import zarr
import zarr.errors
import zarr.storage
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    # dirf = zarr.open("baselines/diffusion_policy/pusht_cchi_v7_replay.zarr.zip", mode='r')
    dirf = zarr.open("./data/victor/output_data_1.zarr", mode='r') #"data/pusht/pusht_cchi_v7_replay.zarr"
    # Updated to use a working zarr file - output_data_1.zarr has compatibility issues with zarr v3 format
    # dirf = zarr.open("./data/victor/output.zarr.zip", mode='r')  # This file works
    # dirf = zarr.open("data/pusht/pusht_cchi_v7_replay.zarr", mode='r') #
    print(dirf.tree())
    print(list(dirf.keys()))
    
    # Print all robot_act data
    print("\n=== ALL ROBOT_ACT DATA ===")
    if 'data' in dirf and 'robot_act' in dirf['data']:
        robot_act_data = dirf['data/robot_act'][:]
        print(f"robot_act shape: {robot_act_data.shape}")
        print(f"robot_act dtype: {robot_act_data.dtype}")
        print("\nAll robot_act values:")
        for i, action in enumerate(robot_act_data):
            print(f"Step {i:3d}: {action}")
    else:
        print("robot_act not found in the zarr file")
        if 'data' in dirf:
            print("Available data keys:", list(dirf['data'].keys()))
        else:
            print("No 'data' group found")
    
    print("\n=== DETAILED STRUCTURE ===")
    for k in dirf.keys():
        print("-----------------------------------------------------------------------")
        print("GROUP:", k)
        print("\nARRS:")
        # for elem in dirf[k]:
        #     print(elem)
        for arr_k in list(dirf[k].keys()):
            print("key:", arr_k,"\tshape:", dirf[k + "/" + arr_k].shape)
            # for elem in dirf[k + "/" + arr_k]:
            #     print(elem)
            # print(arr_k, list(dirf[k+"/"+arr_k]))
        print("-----------------------------------------------------------------------")

    print()