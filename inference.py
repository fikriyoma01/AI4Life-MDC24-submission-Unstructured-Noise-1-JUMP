"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm | gzip -c > example-algorithm.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

from pathlib import Path

import numpy as np
import torch
import SimpleITK
import tifffile


# Name of the expected input and output folders. CHANGE depending on the dataset.
INPUT_PATH = Path("/input/images/image-stack-unstructured-noise/")
OUTPUT_PATH = Path("/output/images/image-stack-denoised/")

# Path to the resource containing YOUR model. See 'src/create_model.py' for an example.
MODEL_PATH = Path("resources/model.pth")


def show_torch_cuda_info():
    """Print cuda information, so it can be availiable in the container logs"""
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)
    print("\n")


def save_result_image_mha(image_array: np.ndarray, result_path: Path):
    """Save the result denoised image.
    Be careful to save results only in .mha format!
    Otherwise, the container will fail"""
    print(f"Writing image to: {result_path}")
    mha_image = SimpleITK.GetImageFromArray(image_array)
    SimpleITK.WriteImage(mha_image, result_path, useCompression=True)


def save_result_image_tiff(image_array: np.ndarray, result_path: Path):
    print(f"Writing an image to: {result_path}")
    with tifffile.TiffWriter(result_path) as out:
        out.write(
            image_array,
            resolutionunit=2
        )


def read_image(image_path: Path) -> (torch.Tensor, np.ndarray):
    """Read input noisy image from image_path"""
    print(f"Reading image: {image_path}")
    input_array = tifffile.imread(image_path)
    input_array = input_array.astype(np.float32)
    print(f"Loaded image shape: {input_array.shape}")
    original_shape = input_array.shape
    # For this example, we will flatten the samples and channels to predict images one by one
    # input_array = input_array.reshape(
    #     (-1, input_array.shape[-2], input_array.shape[-1])
    # )
    input_tensor = torch.from_numpy(input_array)
    print(f"Final input shape: {input_tensor.shape}")
    return input_tensor, original_shape


def main():
    show_torch_cuda_info()

    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    input_files = sorted(INPUT_PATH.glob(f"*.tif*"))
    print(f"Found files: {input_files}")

    print(f"Loading model: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(MODEL_PATH).to(device)
    model.eval()

    for input_file in input_files:
        input_tensor, original_shape = read_image(input_file)
        input_tensor = input_tensor.to(device)

        print("Running inference...")
        result = np.zeros_like(input_tensor.cpu().numpy(), dtype=np.float32)

        # Process each image independently
        for i in range(input_tensor.shape[0]):
            for j in range(input_tensor.shape[1]):
                single_image_tensor = input_tensor[i, j, :, :].unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    denoised_image = model(single_image_tensor).squeeze().cpu().numpy()
                result[i, j, :, :] = denoised_image

        print(f"Output shape: {result.shape}")

        output_path = OUTPUT_PATH / f"{input_file.stem}.tif"
        save_result_image_tiff(image_array=result, result_path=output_path)

if __name__ == "__main__":
    main()