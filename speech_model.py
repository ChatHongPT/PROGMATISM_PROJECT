import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    # First LSTM layer with Batch Normalization and Dropout
    model.add(LSTM(256, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Second LSTM layer with Batch Normalization and Dropout
    model.add(LSTM(256, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Third LSTM layer without return sequences
    model.add(LSTM(256))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Fully connected layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output layer
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model