# This code provides the functions to create, train, and save different models of neural networks

from ModelHandling import prediction_models


def Create_NN(data_path, name, num_of_labels, instruments=False):
    '''
    Create a Basic Neural Network with the num_of_labels representing the number of output nodes, and the data being
    processed from the data_path

    :param data_path: path with the data
    :param name: name to store the file to
    :param num_of_labels: number of labels and therefore number of output nodes
    :param instruments: Is the data instrument data?
    :return:
    '''

    basic_NN = prediction_models.Basic_ML(num_of_labels, data_path)
    if instruments:
        basic_NN.load_data_instruments()
    else:
        basic_NN.load_data()
    basic_NN.create_model()
    basic_NN.compile()
    basic_NN.plot()
    basic_NN.save_model(name)


def Create_CNN(data_path, name, num_of_labels, instruments=False):
    '''
    Creates/Trains a Convolutional Neural Network on the data in data_path, compiles it, plots the accuracy and loss,
    then saves the model
    :param data_path: path with the data
    :param name: name to store the file to
    :param num_of_labels: number of labels and therefore number of output nodes
    :param instruments: Is the data instrument data?
    :return:
    '''

    CNN = prediction_models.CNN(num_of_labels, data_path)
    if instruments:
        CNN.load_data_instruments()
    else:
        CNN.load_data()
    CNN.prepare_data()
    CNN.create_model()
    CNN.compile()
    CNN.plot()
    # CNN.save_model(name)


def Create_RNN(data_path, name, num_of_labels, instruments=False):
    '''
    Creates/Trains a Reccurent Neural Network on the data in data_path, compiles it, plots the accuracy and loss,
    then saves the model
    :param data_path: path with the data
    :param name: name to store the file to
    :param num_of_labels: number of labels and therefore number of output nodes
    :param instruments: Is the data instrument data?
    :return:
    '''
    RNN = prediction_models.RNN(num_of_labels, data_path)
    if instruments:
        RNN.load_data_instruments()
    else:
        RNN.load_data()
    RNN.prepare_data()
    RNN.create_model()
    RNN.compile()
    RNN.plot()
    RNN.save_model(name)


# The commented out code for calling the 'Create' (RNN, CNN and NN) functions


# data_path = r'E:\NEA_Files\Training_Data_Genre.json.txt'
# Create_NN(data_path, 'genre_NN', 10)
# Create_RNN(data_path, 'genre_RNN', 10)
# Create_CNN(data_path, 'genre_CNN', 10)

data_path = r'E:\NEA_Files\Training_Data_Instruments'
# Create_NN(data_path, 'instruments_NN', 20, True)
# Create_RNN(data_path, 'instruments_RNN', 20, True)
Create_CNN(data_path, 'instruments_CNN', 20, True)
