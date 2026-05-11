from PIL import Image
import sys 
import os

def convert_jpeg_to_png(input_image, target_dir=None):
    jpeg_image = Image.open(input_image)

    target_dir = os.path.dirname(input_image) if target_dir is None else target_dir
    image_name = os.path.basename(input_image)
    # Save as PNG
    jpeg_image.save(os.path.join(target_dir, image_name.replace('.jpeg', '_original.png')))


if __name__ == '__main__':
    # Open the JPEG image
    input_image = sys.argv[1]
    convert_jpeg_to_png(input_image)
