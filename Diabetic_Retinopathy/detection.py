from tensorflow.keras.preprocessing import image
import numpy as np


def load_weight(model, weights_path):
    model.load_weights(weights_path)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img = img_array / 255
    img = img.reshape(-1,256,256,3)
    return img


def detection(model, img_array):
    prediction = model.predict(img_array)
    predict = np.argmax(prediction)
    return predict