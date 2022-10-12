"""
Generic setup of the data sources and the model training. 

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on 
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""
import tensorflow as tf
from tensorflow.keras import backend as Kc
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint


from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pickle import dump,load
import logging


# Helper: Early stopping.
early_stopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto' )

#monitor='val_loss',patience=2,verbose=0
#In your case, you can see that your training loss is not dropping - which means you are learning nothing after each epoch. 
#It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.

#Cree esta funcion para obtener los datos que necesito de mi dataset
def get_del_dataset():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes  = 2 #dataset dependent 
    batch_size  = 64
    epochs      = 50
    entrada=load( open('columnas.pkl', 'rb'))
    input_shape=(entrada,)
    y=load( open('labels.pkl', 'rb'))
    x=load( open('data.pkl', 'rb'))
    
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    xnorm = scaler.fit_transform(x)
    
    
    x_train, x_test, y_train, y_test = train_test_split(xnorm,y,test_size=0.20, random_state=33)
    
    #Exportamos estos datos para utilizarlos en el entrenamiento de nuestra red
    dump(x_train, open('x_train.pkl', 'wb'))
    dump(x_test, open('x_test.pkl', 'wb'))
    dump(y_train, open('y_trainLR.pkl', 'wb'))
    dump(y_test, open('y_testLR.pkl', 'wb'))
    
    print(np.shape(y_train))
    print(np.shape(x_train))
    
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test, nb_classes)
    
    dump(y_train, open('y_train.pkl', 'wb'))
    dump(y_test, open('y_test.pkl', 'wb'))
    
    print(np.shape(y_train))
    print(np.shape(x_train))
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def compile_model_dataset(genome, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers  = genome.geneparam['nb_layers' ]
    nb_neurons = genome.nb_neurons()
    activation = genome.geneparam['activation']
    optimizer  = genome.geneparam['optimizer' ]

    logging.info("Architecture:%s,%s,%s,%d" % (str(nb_neurons), activation, optimizer, nb_layers))

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons[i], activation='softmax', input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons[i], activation='softmax'))

        model.add(Dropout(0.2))  # hard-coded dropout for each layer

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

precision=0.0 
 
def train_and_score(genome, dataset): 
    """Train the model, return test loss. 
 
    Args: 
        network (dict): the parameters of the network 
        dataset (str): Dataset to use for training/evaluating 
 
    """ 
    global precision
    
    logging.info("Getting Keras datasets") 
 
    nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_del_dataset() 
         
    logging.info("Compling Keras model") 
 
    model=compile_model_dataset(genome, nb_classes, input_shape) 
         
    history = LossHistory() 
     
    print(np.shape(x_train)) 
    print(np.shape(y_train)) 
     
    model.fit(x_train, y_train, 
              batch_size=batch_size, 
              epochs=epochs,   
              # using early stopping so no real limit - don't want to waste time on horrible architectures 
              verbose=1, 
              validation_data=(x_test, y_test), 
              #callbacks=[history]) 
              callbacks=[early_stopper]) 
 
    score = model.evaluate(x_test, y_test, verbose=0) 
     
    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1]) 
     
    if score[1]> precision: 
        print("\n Modelo mas eficiente creado \n") 
        model.save('modelo_guardado.h5') 
        precision=score[1] 
     
         
     
    Kc.clear_session() 
    #we do not care about keeping any of this in memory -  
    #we just need to know the final scores and the architecture 
     
    return score[1]  # 1 is accuracy. 0 is loss.