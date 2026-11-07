
import h5py

file_path = r"C:\Users\tharu\Downloads\nyu_depth_v2_labeled (1).mat"

print("Opening NYU Depth V2 dataset...")

try:
    with h5py.File(file_path, "r") as f:
        print("\nDataset opened successfully!")
        print("\nAvailable keys:")

        for key in f.keys():
            item = f[key]
            print(f"\nName: {key}")

            if isinstance(item, h5py.Dataset):
                print("Shape:", item.shape)
                print("Data type:", item.dtype)
            else:
                print("Group:", list(item.keys())[:10])

except Exception as e:
    print("Could not open using HDF5.")
    print("Error:", e)
