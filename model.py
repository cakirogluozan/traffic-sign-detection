
import keras

from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.applications.mobilenet import MobileNet

def fcn(weight_dir):
    inputs = Input(shape=(1024,))

    # a layer instance is callable on a tensor, and returns a tensor
    output_1 = Dense(64, activation='relu')(inputs)
    output_2 = Dense(64, activation='relu')(output_1)
    predictions = Dense(4, activation='softmax')(output_2)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.load_weights(weight_dir)
    return model

def lenet(weight_dir, input_shape, num_classes):
    inputs = Input(shape=input_shape)

    conv_out = Conv2D(6, (3, 3), activation='relu')(inputs)
    avg_out  = AveragePooling2D()(conv_out)
    conv_out2 = Conv2D(6, (3, 3), activation='relu')(avg_out)
    avg_out2  = AveragePooling2D()(conv_out2)
    flatten_out = Flatten()(avg_out2)
    dense_out = Dense(units=120, activation='relu')(flatten_out)
    dense_out2 = Dense(units=84, activation='relu')(dense_out)
    outputs = Dense(units=num_classes, activation='softmax')(dense_out2)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.load_weights(weight_dir)
    return model


def mobilenet(weight_dir):
    model = MobileNet(input_shape=(64,64,1), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights= None, input_tensor=None, pooling='avg', classes=4)
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    
    model.load_weights(weight_dir)
    return model
