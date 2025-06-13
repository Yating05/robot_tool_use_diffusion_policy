import zarr
import zarr.errors
import zarr.storage
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    # dirf = zarr.open("baselines/diffusion_policy/pusht_cchi_v7_replay.zarr.zip", mode='r')
    dirf = zarr.open("data/victor/d1.zarr", mode='r') #"data/pusht/pusht_cchi_v7_replay.zarr"
    # dirf = zarr.open("data/pusht/pusht_cchi_v7_replay.zarr", mode='r') #
    print(dirf.tree())
    print(list(dirf.keys()))
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