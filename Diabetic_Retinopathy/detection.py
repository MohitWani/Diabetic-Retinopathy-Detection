import numpy as np
from PIL import Image

def load_weight(model, weights_path):
    model.load_weights(weights_path)

def preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img = img_array / 255
    img = img.reshape(-1,256,256,3)
    return img


def detection(model, img_array):
    prediction = model.predict(img_array)
    predict = np.argmax(prediction)
    return predict