from PIL import Image
import os

# Function to crop images in a folder and save them to another folder
def crop_and_save_images(input_folder, output_folder, target_width, target_height):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for i, filename in enumerate(os.listdir(input_folder)):
        # Check if the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Open the image file
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            
            # Crop the image to the desired size
            cropped_image = image.crop((0, 0, target_width, target_height))
            
            # Save the cropped image to the output folder with a sequential name
            output_filename = f"{i+1+81}.png"  # Change the extension as per your requirement
            output_path = os.path.join(output_folder, output_filename)
            cropped_image.save(output_path)
            print(f"{filename} cropped and saved as {output_filename}")

# Provide the input and output folder paths, and target size
input_folder = "images\\standard\\"
output_folder = "images\\cropped\\"
target_width = 6224
target_height = 1024

# Call the function to crop and save images
crop_and_save_images(input_folder, output_folder, target_width, target_height)
