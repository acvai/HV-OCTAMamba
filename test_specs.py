import torch

def get_gpu_specs():
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        print("CUDA is available. GPU detected!")
    else:
        print("No GPU detected. Using CPU.")
        return

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Retrieve and print GPU specifications
    for i in range(num_gpus):
        print(f"\nGPU {i} Specs:")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        # print(f"Multiprocessor Count: {torch.cuda.get_device_properties(i).multi_processor_count}")
        # print(f"Max Threads per Block: {torch.cuda.get_device_properties(i).max_threads_per_block}")
        # print(f"Max Threads per Multiprocessor: {torch.cuda.get_device_properties(i).max_threads_per_multiprocessor}")
        # print(f"Max Grid Size: {torch.cuda.get_device_properties(i).max_grid_size}")
        # print(f"Clock Rate: {torch.cuda.get_device_properties(i).clock_rate / 1e6:.2f} GHz")

    # Set the default GPU (optional)
    torch.cuda.set_device(0)
    print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    get_gpu_specs()