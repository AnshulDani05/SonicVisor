import prediction_models


def Load_NN(num_of_labels):
    '''
    Load the Basic Neural Network for instruments
    :param num_of_labels: Number of labels
    :return:
    '''
    basic_NN = prediction_models.Basic_ML(num_of_labels)
    basic_NN.load_model('instruments_NN.pkl')
    return basic_NN


def Load_CNN(num_of_labels):
    '''
    Load the Convolutional Neural Network for instruments
    :param num_of_labels: Number of Labels
    :return:
    '''
    CNN = prediction_models.CNN(num_of_labels)
    CNN.load_model('instruments_CNN.pkl')
    return CNN


def Load_RNN(num_of_labels):
    '''
    Load the Reccurent Neural Network for instruments
    :param num_of_labels: Number of Labels
    :return:
    '''
    RNN = prediction_models.RNN(num_of_labels)
    RNN.load_model('instruments_RNN.pkl')
    return RNN
