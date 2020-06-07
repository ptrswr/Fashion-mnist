from tensorflow import keras
from keras.datasets import fashion_mnist


def processTensor():

    ((X_train, y_train), (X_test, y_test)) = fashion_mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), padding='same',
                            activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2), strides=2),

        keras.layers.Conv2D(64, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2), strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train,
              y_train,
              batch_size=64,
              epochs=10)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)


if __name__ == "__main__":
    # execute only if run as a script
    processTensor()
