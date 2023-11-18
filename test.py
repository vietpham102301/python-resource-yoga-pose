# from PIL import Image
# import imghdr
#
# file_path = "F:\\saved_frames\\2909_1699202536.png"
# try:
#     # image = Image.open(file_path)
#
#     print(imghdr.what(file_path))
#     # You can also access other image properties if needed, e.g., image.size, image.mode, etc.
# except:
#     print("The file is not a valid PNG.")


import cv2
import json

def draw_rectangle_on_image(input_image_path, output_image_path, bounds):
    # Load the input image
    image = cv2.imread(input_image_path)

    # Extract the bounds
    x = int(bounds["x"])
    y = int(bounds["y"])
    width = int(bounds["w"])
    height = int(bounds["h"])

    # Draw a rectangle on the image
    color = (0, 0, 255)  # Red color (BGR format)
    thickness = 2
    cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)

    # Save the output image
    cv2.imwrite(output_image_path, image)

if __name__ == "__main__":
    # Input image path
    input_image_path = r"F:\saved_frames\9005_1700067486.png"  # Replace with your image file path

    # Output image path
    output_image_path = r"F:/saved_frames/result_test.jpg"  # Replace with the desired output file path

    # Bounds as a dictionary
    bounds_json = {'x': 65, 'y': 0, 'w': 632, 'h': 599}

    # Draw the rectangle on the image
    draw_rectangle_on_image(input_image_path, output_image_path, bounds_json)

