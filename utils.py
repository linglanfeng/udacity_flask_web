import os
from keras.models import load_model

def get_best_model(model_name):
    """
    Defines the model
    :return: Returns the model
    """
    """
    Check if a model already exists
    """
    if os.path.exists('./train_best_model/best_{}_model.hdf5'.format(model_name)):
        model = load_model('./train_best_model/best_{}_model.hdf5'.format(model_name))
        print('Best model fetched from the disk')
        model.summary()

    return model
