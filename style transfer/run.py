"""
John Jefferson III
Daniel Rodriguez
Michael Patel

November 2020

Project description:
    Use ML style transfer to generate digital art

File description:
    Loads and generates images
"""
################################################################################
# Imports
import os
import numpy as np
from PIL import Image
from datetime import datetime

import tensorflow as tf
import tensorflow_hub as hub


################################################################################
# directories
DATA_DIR = os.path.join(os.getcwd(), "data")


################################################################################
# load image
def load_image(filename, mode):
    data_subdirectory = os.path.join(DATA_DIR, mode)
    filepath = os.path.join(data_subdirectory, filename)
    image = Image.open(filepath)
    return image


# preprocess image
def preprocess_image(filename, mode="content"):
    image = load_image(filename, mode)

    if mode is "style":
        # style image size should be 256x256
        # content image size can be arbitrary
        image = image.resize((256, 256))

    # image to array
    image = tf.keras.preprocessing.image.img_to_array(image)

    # normalize image: [0, 255] --> [0, 1]
    image = image / 255.0

    # reshape: (1, WIDTH, HEIGHT, CHANNELS)
    # currently, batch size = 1
    image = np.expand_dims(image, axis=0)

    return image


################################################################################
# Main
if __name__ == "__main__":
    # environment setup
    print(f'TF version: {tf.__version__}')
    print(f'TF hub version: {hub.__version__}')
    print(f'GPU available: {tf.test.is_gpu_available()}')

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    # Content image
    content_image_filename = "content.jpg"
    content_image = preprocess_image(
        filename=content_image_filename,
        mode="content"
    )

    # Style image
    style_image_filename = "style_pencil_drawn.jpg"
    style_image = preprocess_image(
        filename=style_image_filename,
        mode="style"
    )

    # ---- MODEL ---- #
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    # ----- GENERATE ----- #
    outputs = hub_module(
        tf.constant(content_image),
        tf.constant(style_image)
    )
    #print(f'Num outputs: {len(outputs)}')
    output_image = outputs[0]

    # reshape: (WIDTH, HEIGHT, CHANNELS)
    output_image = tf.squeeze(output_image, axis=0)
    #output_image = output_image * 255.0

    # tensor to image
    output_image = tf.keras.preprocessing.image.array_to_img(output_image)
    #print(output_image)
    #output_image.show()

    # save generated image
    output_image_filename = "generated_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".jpg"
    output_image_subdirectory = os.path.join(DATA_DIR, "generated")
    output_image_filepath = os.path.join(output_image_subdirectory, output_image_filename)
    output_image.save(output_image_filepath)
