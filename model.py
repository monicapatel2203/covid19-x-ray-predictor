import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Flatten, Dense

def load_model(model_weights):
    model_3 = keras.Sequential()
    #Convolutional Layer
    model_3.add(Conv2D(filters=16, kernel_size = 3, input_shape = (256,256,1), strides=2, padding = "same", activation="relu"))

    #Convolutional Layer
    model_3.add(Conv2D(filters=64, kernel_size = 3, strides=2,  padding = "same", activation="relu"))

    #Convolutional Layer
    model_3.add(Conv2D(filters=128, kernel_size = 3, strides=2,  padding = "same", activation="relu"))

    #Convolutional Layer
    model_3.add(Conv2D(filters=256, kernel_size = 3, strides=2,  padding = "same", activation="relu"))

    #Flatten Layer
    model_3.add(Flatten())

    #Fully connected layer
    model_3.add(Dense(128,activation="relu"))
    model_3.add(Dense(128,activation="relu"))

    #Output Layer
    model_3.add(Dense(3,activation="softmax"))

    model_3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model_3.load_weights(model_weights)

    return model_3

if __name__ == "__main__":
    model = load_model("./model3.keras")

    print(model.summary())