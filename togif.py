from PIL import Image
import os

# Directory where your image files are located
image_directory = "output"

# List of image file names
image_files = os.listdir(image_directory)
duration = 100

frames = [Image.open(image_name) for image_name in image_files]

frames[0].save("output.gif", save_all=True, append_images=frames[1:], duration=duration, loop=0)
