import tensorflow as tf
from tensorflow.keras.preprocessing import image


def detection(img_path):
    model = tf.keras.models.load_model('retina_weights.hdf5')
    img = image.load_img(img_path, target_size=(224, 224))
    prediction = model.predict(img)
    return prediction