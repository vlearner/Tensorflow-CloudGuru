import tensorflow as ts
import tensorflow_datasets as tfds

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def wrangle_data(dataset, split):
    wrangled = dataset.map(lambda img, lbl: (ts.cast(img, ts.float32) / 255.0, lbl))
    wrangled = wrangled.cache()
    if split == 'train':
        wrangled = wrangled.shuffle(60000)
    return wrangled.batch(64).prefetch(ts.data.AUTOTUNE)


def compile_model(new_model):
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    return new_model


def create_model():
    new_model = ts.keras.Sequential([
        ts.keras.layers.InputLayer((28, 28, 1)),
        ts.keras.layers.Flatten(),
        ts.keras.layers.Dense(64, activation='relu'),
        ts.keras.layers.Dense(10, activation='softmax'),
    ])
    return compile_model(new_model)


if __name__ == '__main__':
    mnist_train, info = tfds.load('mnist', split='train', as_supervised=True, with_info=True)
    mnist_test = tfds.load('mnist', split='test', as_supervised=True)

    train_data = wrangle_data(mnist_train, 'train')
    mnist_test = wrangle_data(mnist_test, 'test')

    model = create_model()

    history = model.fit(train_data, epochs=5)

    model.save('mnist.keras')
