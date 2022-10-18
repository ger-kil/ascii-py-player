from pathlib import Path
from typing import List
import numpy as np
import cv2

#ASCII_LIGHT_SCALE = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,^`'. "
ASCII_LIGHT_SCALE = "$@%&#*/\|(){}[]?-_+~<>!;:,^`'. "
PIXEL = 100


def transform_image(image_path: Path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img


def rescale_image(image: List, width: int = None, height: int = None, interpol = cv2.INTER_AREA):

    # grab initial image size
    (original_height, original_width) = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is not None and height is not None:
        dim = (height, width)

    elif width is None:
        ratio = height / float(original_height)
        dim = (int(original_width * ratio), height)
    
    else:
        ratio = width / float(original_width)
        dim = (width, int(original_height * ratio))

    rescaled_image = cv2.resize(image, dim, interpolation=interpol)
    
    return rescaled_image


def ascii_transformer(grey_image: List, chars: str):

    #ascii_matrix = np.

    char_index_matrix = np.interp(grey_image, (grey_image.min(), grey_image.max()), (0, len(chars) - 1))
    char_index_matrix = np.rint(char_index_matrix)
    print(char_index_matrix)

    ascii_matrix = char_index_matrix.astype(str)

    for i, row in enumerate(char_index_matrix):
        for j, entry in enumerate(row):
            ascii_matrix[i][j] = chars[int(entry)]

    return ascii_matrix



def print_ascii_image(ascii_image):
    for row in ascii_image:
        print("".join(row))
    return

if __name__ == '__main__':
    transformed_img = transform_image('dog_image.jpg')
    print(transformed_img)
    print(transformed_img.shape)


    rescaled_img = rescale_image(transformed_img, PIXEL)
    print(rescaled_img)
    print(rescaled_img.shape)

    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.imshow("Resized_Window", rescaled_img)
    cv2.waitKey(0)

    ascii_image = ascii_transformer(rescaled_img, ASCII_LIGHT_SCALE)

    print_ascii_image(ascii_image)