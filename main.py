import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from skimage import io
import os
import keras
import keras.layers as L
import skimage.transform


def show_image(x):
    plt.imshow(x, cmap=plt.get_cmap('gray'))


def visualize(img, encoder, decoder):
    """Draws original, encoded and decoded images"""
    code = encoder.predict(img[None])[0]  # img[None] is the same as img[np.newaxis, :]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1, 3, 2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1] // 2, -1]))

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()


def load_set():
    return np.array(list(map(
        lambda x: skimage.transform.resize(io.imread(os.path.join('pokemon_set/images', x), as_gray=True), (64, 64),
                                           mode='edge',
                                           anti_aliasing=False,
                                           anti_aliasing_sigma=None,
                                           preserve_range=True,
                                           order=0),
        os.listdir('pokemon_set/images'))))


def build_autoencoder(img_shape, code_size):
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Conv2D(32, (3, 3), (1, 1), padding="same", activation="elu"))
    encoder.add(L.MaxPooling2D((2, 2)))
    encoder.add(L.Conv2D(64, (3, 3), (1, 1), padding="same", activation="elu"))
    encoder.add(L.MaxPooling2D((2, 2)))
    encoder.add(L.Conv2D(128, (3, 3), (1, 1), padding="same", activation="elu"))
    encoder.add(L.MaxPooling2D((2, 2)))
    encoder.add(L.Conv2D(256, (5, 5), (1, 1), padding="same", activation="elu"))
    encoder.add(L.MaxPooling2D((2, 2)))
    encoder.add(L.Conv2D(512, (5, 5), (1, 1), padding="same", activation="elu"))
    encoder.add(L.MaxPooling2D((2, 2)))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size))

    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(2 * 2 * 512))
    decoder.add(L.Reshape((2, 2, 512)))
    decoder.add(L.Conv2DTranspose(filters=256, kernel_size=(5, 5), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, activation=None, padding='same'))

    return encoder, decoder


img_set = load_set()
print(img_set.shape)
IMG_SHAPE = (64, 64, 1)

plt.title('sample images')

for i in range(6):
    plt.subplot(2, 3, i + 1)
    show_image(img_set[i + 200])  # Showing some images
plt.show()
encoder, decoder = build_autoencoder(IMG_SHAPE, 32)
encoder.summary()
decoder.summary()
X_train, X_test = train_test_split(img_set, test_size=0.15, random_state=47)

inp = L.InputLayer(IMG_SHAPE)
code = encoder(inp)
reconstructed = decoder(code)

autoencoder = keras.models.Model(inputs=inp, outputs=reconstructed)
autoencoder.compile(optimizer="adamax", loss='mse')

autoencoder.fit(x=X_train, y=X_train, epochs=25,
                validation_data=[X_test, X_test],
                verbose=1,
                initial_epoch=0)  #
