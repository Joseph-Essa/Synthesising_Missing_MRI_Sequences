import os
import nibabel as nib
import shutil
import numpy as np
from imageio import imwrite
import cv2


def extract_middle_slices(nii_path, output_dir, num_slices=20):
    # Load the .nii image
    img = nib.load(nii_path)
    img_data = img.get_fdata()
    
    # Get the middle slice index
    middle_index = img_data.shape[2] // 2
    
    # Calculate the start and end indices for the middle slices
    start_index = middle_index - (num_slices // 2)
    end_index = start_index + num_slices
    
    # Extract the middle slices
    middle_slices = img_data[:, :, start_index:end_index]
    
    # Create the output file path
    file_name = os.path.basename(nii_path)
    output_path = os.path.join(output_dir, f"middle_slices_{file_name}")
    
    # Save the middle slices as a new .nii file
    middle_img = nib.Nifti1Image(middle_slices, img.affine)
    nib.save(middle_img, output_path)

def process_folders(root_folder, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Traverse the directory tree
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('t2w.nii.gz'):
                file_path = os.path.join(dirpath, filename)
                extract_middle_slices(file_path, output_folder)

# Define the root folder containing the subfolders with .nii files and the output folder
root_folder = r"#"
output_folder = r"#"


# Process the folders
process_folders(root_folder, output_folder)

def extract_nii_files(source_folder, destination_folder, suffix='t2w'):
    """
    Traverse through the source_folder, find .nii files ending with the specified suffix, 
    and copy them to the destination_folder.
    
    Args:
    source_folder (str): The root folder containing subfolders with .nii files.
    destination_folder (str): The folder where the extracted .nii files will be copied.
    suffix (str): The suffix that the .nii files should end with. Default is 't1c'.
    """
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # Traverse the directory tree
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.nii.gz') and file.endswith(suffix + '.nii.gz'):
                # Construct full file path
                file_path = os.path.join(root, file)
                # Copy the file to the destination folder
                shutil.copy2(file_path, destination_folder)
                print(f"Copied: {file_path} to {destination_folder}")

# Example usage:
source_folder = r"#"
destination_folder = r"#"
extract_nii_files(source_folder, destination_folder)


def extract_middle_slices_nii_to_png(input_folder, output_folder):
    # Create output directory if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all .nii files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            filepath = os.path.join(input_folder, filename)
            nii = nib.load(filepath)
            data = nii.get_fdata()

            # Determine the middle 20 slices along the third axis
            num_slices = data.shape[2]
            middle_slices_start = num_slices // 2 - 10
            middle_slices_end = num_slices // 2 + 10

            for i in range(middle_slices_start, middle_slices_end):
                slice_data = data[:, :, i]
                # Normalize the slice data to [0, 255]
                slice_data = 255 * (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
                slice_data = slice_data.astype(np.uint8)

                # Create a filename for the PNG image
                output_filename = f"{os.path.splitext(filename)[0]}_slice_{i}.png"
                output_filepath = os.path.join(output_folder, output_filename)
                
                # Save the slice as a PNG image
                imwrite(output_filepath, slice_data)
                print(f"Saved {output_filepath}")

# Example usage:
input_folder = r"#"
output_folder = r"#"
extract_middle_slices_nii_to_png(input_folder, output_folder)


def crop_black_space(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find all non-black pixels
    coords = cv2.findNonZero(gray)
    
    # Check if there are any non-black pixels found
    if coords is None:
        # If no non-black pixels are found, the image is considered completely black
        #print("Image is completely black.")
        return None
    
    # Create a bounding box around the non-black pixels
    x, y, w, h = cv2.boundingRect(coords)
    
    # Check if the bounding box has a valid size
    if w == 0 or h == 0:
        #print("Invalid bounding box size, returning original image")
        return image
    
    # Crop the image to the bounding box
    cropped = image[y:y+h, x:x+w]
    return cropped

def process_images(input_folder, output_folder):
    # # Create the output folder if it doesn't exist
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Read the image
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            
            # Check if the image was successfully loaded
            if image is None:
                #print(f"Warning: Could not read image {img_path}. Skipping.")
                continue
            
            # Crop the black space
            cropped_image = crop_black_space(image)
            
            # If the image is completely black, delete it
            if cropped_image is None:
                #os.remove(img_path)
                #print(f"Deleted completely black image {img_path}.")
                continue
            
            # Save the cropped image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_image)

    print("Phase 2 Done")



input_folder =  r"#"
output_folder = r"#"

process_images(input_folder, output_folder)