"""
John Jefferson III
Daniel Rodriguez
Michael Patel

November 2020

Project description:
    Use ML style transfer to generate digital art

File description:
    Loads and generates images using a Flask server
"""
################################################################################
# Imports
import os
import numpy as np
from PIL import Image
from datetime import datetime
import shutil
import glob
import time
import socket

import tensorflow as tf
import tensorflow_hub as hub

from flask import Flask, render_template, request
app = Flask(__name__)
UPLOAD_PATH = os.path.join(os.getcwd(), "style transfer\\static\\uploads")
app.config['UPLOAD_PATH'] = UPLOAD_PATH
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


################################################################################
# directories
DATA_DIR = os.path.join(os.getcwd(), "style transfer\\data")


################################################################################
# load image
def load_image(filename, mode):
    #data_subdirectory = os.path.join(DATA_DIR, mode)
    data_subdirectory = os.path.join(os.getcwd(), "style transfer\\static\\uploads")
    filepath = os.path.join(data_subdirectory, filename)

    """
    if "content" in filename or "style" in filename:
        filename = filename.split("_")[1]  # split @ underscore
    name = str(filename.split(".")[0])  # split @ .extension
    """
    name = filename
    image = Image.open(filepath)
    return image, name


# preprocess image
def preprocess_image(filename, mode="content"):
    image, name = load_image(filename, mode)

    if mode == "content":
        if image.width > 2048:
            if image.height > 1024:
                image = image.resize((2048, 1024))
            else:
                image = image.resize((2048, image.height))
        else:
            if image.height > 1024:
                image = image.resize((image.width, 1024))

    if mode == "style":
        # style image size should be 256x256
        # content image size can be arbitrary
        image = image.resize((256, 256))

    # convert PIL image to numpy array
    image = tf.keras.preprocessing.image.img_to_array(image)

    # normalize image: [0, 255] --> [0, 1]
    image = image / 255.0

    # reshape: (1, WIDTH, HEIGHT, CHANNELS)
    # currently, batch size = 1
    image = np.expand_dims(image, axis=0)

    # convert numpy array to tensor
    image = tf.convert_to_tensor(image)

    return image, name


# generate output image
def generate_image(content_image_filename, style_image_filename):
    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    # content image
    content_image, content_name = preprocess_image(
        filename=content_image_filename,
        mode="content"
    )

    # style image
    style_image, style_name = preprocess_image(
        filename=style_image_filename,
        mode="style"
    )

    # ---- MODEL ---- #
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    model = hub.load(hub_handle)

    # ----- GENERATE ----- #
    outputs = model(
        content_image,
        style_image
    )

    # print(f'Num outputs: {len(outputs)}')
    output_image = outputs[0]

    # reshape: (WIDTH, HEIGHT, CHANNELS)
    output_image = tf.squeeze(output_image, axis=0)

    # [0, 1] --> [0, 255]
    #output_image = output_image * 255.0

    # convert numpy array to PIL image
    output_image = tf.keras.preprocessing.image.array_to_img(output_image)
    # print(output_image)
    # output_image.show()

    # save generated image
    #output_image_filename = "generated_" + content_name + "_" + style_name + ".jpg"
    output_image_filename = "generated.jpg"
    # output_image_filename = "generated_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".jpg"
    output_image_subdirectory = os.path.join(DATA_DIR, "generated")
    if not os.path.exists(output_image_subdirectory):
        os.makedirs(output_image_subdirectory)

    output_image_filepath = os.path.join(output_image_subdirectory, output_image_filename)
    output_image.save(output_image_filepath)


################################################################################
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/Upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    else:
        t = os.path.join(os.getcwd(), "style transfer\\static\\generated.jpg")
        if os.path.exists(t):
            os.remove(t)

        # content
        content = request.files["content"]
        content_path = os.path.join(app.config['UPLOAD_PATH'], 'content.jpeg')
        content.save(content_path)

        # style
        style = request.files["style"]
        style_path = os.path.join(app.config['UPLOAD_PATH'], 'style.jpeg')
        style.save(style_path)

        # generate new picture
        generate_image(
            content_image_filename="content.jpeg",
            style_image_filename="style.jpeg"
        )

        # copy from generated/ to static/
        shutil.copy(
            src=os.path.join(DATA_DIR, "generated\\generated.jpg"),
            dst=os.path.join(os.getcwd(), "style transfer\\static\\generated.jpg")
        )

        return render_template("home.html")


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


################################################################################
# Main
if __name__ == "__main__":
    # environment setup
    #print(f'TF version: {tf.__version__}')
    #print(f'TF hub version: {hub.__version__}')
    #print(f'GPU available: {tf.test.is_gpu_available()}')

    ip_address = socket.gethostbyname(socket.gethostname())
    app.run(host=ip_address)
