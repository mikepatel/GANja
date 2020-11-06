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
    content_image_filepath = os.path.join(DATA_DIR, content_image_filename)
    content_image = Image.open(content_image_filepath)

    # resize, content image size can be arbitrary
    #content_image = content_image.resize((700, 500))

    # image to array
    content_image = tf.keras.preprocessing.image.img_to_array(content_image)

    # normalize image
    content_image = content_image / 255.0

    # reshape: (1, WIDTH, HEIGHT, CHANNELS)
    content_image = np.expand_dims(content_image, axis=0)

    # Style image
    style_image_filename = "style.jpg"
    style_image_filepath = os.path.join(DATA_DIR, style_image_filename)
    style_image = Image.open(style_image_filepath)

    # style image size should be 256x256
    style_image = style_image.resize((256, 256))

    # image to array
    style_image = tf.keras.preprocessing.image.img_to_array(style_image)

    # normalize image
    style_image = style_image / 255.0

    # reshape: (1, WIDTH, HEIGHT, CHANNELS)
    style_image = np.expand_dims(style_image, axis=0)

    # ---- MODEL ---- #
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    # ----- GENERATE ----- #
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    #print(f'Num outputs: {len(outputs)}')
    output_image = outputs[0]

    # reshape: (WIDTH, HEIGHT, CHANNELS)
    output_image = tf.squeeze(output_image, axis=0)
    #output_image = output_image * 255.0

    # tensor to image
    output_image = tf.keras.preprocessing.image.array_to_img(output_image)
    #print(output_image)
    #output_image.show()
    output_image_filename = "generated_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".jpg"
    output_image_filepath = os.path.join(DATA_DIR, output_image_filename)
    output_image.save(output_image_filepath)
