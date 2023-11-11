# python3
'''
    provide a model for training / predictions
    A model is determined by many factors

'''

from keras.layers import Dense
from keras.models import Sequential
import keras
from sklearn.preprocessing import StandardScaler


def get_model(m_parms=None):
    size = 5
    nfeatures = m_parms.nfeatures
    output = m_parms.output  # output categories
    layers = m_parms.layers  # list of numbers indicate the layer size
    model = Sequential()
    # input layer
    model.add(Dense(nfeatures, input_dim=nfeatures,
        activation='relu', kernel_initializer='he_uniform'))

    # hidden layers
    for size in layers.split():
        size = int(size)
        model.add(Dense(size*nfeatures, activation='relu'))

    #output layer
    model.add(Dense(output, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    return model

