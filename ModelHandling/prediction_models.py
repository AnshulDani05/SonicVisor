import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import pickle
import os
import matplotlib.pyplot as plt


class ML_models:
    def __init__(self, data_path=None, num_of_labels=None):
        '''
        Initialise model variables
        :param data_path:
        :param num_of_labels:
        '''
        self.data_path = data_path
        self.mfcc_input = None
        self.target_instrument = None
        self.history = None
        self.MFCC_train = None
        self.MFCC_test = None
        self.label_train = None
        self.label_test = None
        self.model = None
        self.epochs = None
        self.final_activation = 'softmax'
        self.loss = keras.losses.SparseCategoricalCrossentropy()
        self.test_size = 0.25
        self.validation_size = 0.2
        self.modeltype = None
        self.num_of_labels = num_of_labels

    def plot(self):
        '''
        Plot the history of the compiled model
        :return:
        '''
        history = self.history
        fig, axs = plt.subplots(2)

        axs[0].plot(history.history["accuracy"], label="train accuracy")
        axs[0].plot(history.history["val_accuracy"], label="test accuracy")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend(loc="lower right")
        axs[0].set_title("Accuracy eval")
        axs[1].plot(history.history["loss"], label="train error")
        axs[1].plot(history.history["val_loss"], label="test error")
        axs[1].set_ylabel("Error")
        axs[1].set_xlabel("Epoch")
        axs[1].legend(loc="upper right")
        axs[1].set_title("Error eval")

        plt.show()

        return True

    def predict_model(self, MFCC):
        '''
        Predict the label with the MFCC data
        :param MFCC: data
        :return:
        '''
        MFCC = MFCC.reshape((1, 431, 13))

        prediction = self.model.predict(MFCC)

        print(prediction)

        top_3_indicies = np.argsort(prediction)
        top_3_indicies = (top_3_indicies.tolist())[0]
        top_3_indicies = top_3_indicies[::-1][:3]

        return top_3_indicies

    def load_data(self):
        '''
        Load the data from the json file (for the genres)
        :return:
        '''
        with open(self.data_path, "r") as fp:
            data = json.load(fp)

        mfcc_input = np.array(data['mfccs'])
        target_instrument = np.array([int(num) for num in data["labels"]])

        self.mfcc_input = mfcc_input
        self.target_instrument = target_instrument

        print("Data loaded!")

    def load_data_instruments(self):
        '''
        Load the data from the json file (for the instruments)
        :return:
        '''
        mfccs = []
        instruments = []

        running = True
        while running:
            count = 0
            for filename in os.listdir(self.data_path):
                file_path = os.path.join(self.data_path, filename)
                with open(file_path, 'r') as fp:
                    data = json.load(fp)
                    try:
                        mfccs.append(data[0])
                        instrument_labels = [0 for i in range(self.num_of_labels)]
                        for i in data[1]:
                            instrument_labels[i] = 1
                        instruments.append(instrument_labels)
                    except:
                        print('MEMORY OUT')
                        running = False
                        break
                    count += 1
                    if count > 20000:
                        break
            running = False

        success = False
        while not success:
            try:
                mfccs = np.array(mfccs)
                success = True
            except:
                mfccs = mfccs[:-100]

        instruments = np.array(instruments)
        print(np.shape(mfccs))
        print(np.shape(instruments))

        self.loss = keras.losses.binary_crossentropy  # Change the loss function to binary
        self.final_activation = 'sigmoid'  # Change the final activation to sigmoid

        self.mfcc_input = mfccs
        self.target_instrument = instruments

    def compile(self):
        '''
        Compiles the model and stores the training history
        :return:
        '''
        model = self.model
        optimiser = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimiser,
                      loss=self.loss,
                      metrics=['accuracy'])

        model.summary()
        history = model.fit(self.MFCC_train, self.label_train, validation_data=(self.MFCC_test, self.label_test),
                            batch_size=32,
                            epochs=self.epochs)
        self.history = history

    def save_model(self, name):
        '''
        Save the model to a pickle file
        :param name: Name of the pkl file to store the model to
        :return:
        '''
        with open(f'{name}.pkl', 'wb') as files:
            pickle.dump(self.model, files)
        print('Saved')

    def load_model(self, file_path):
        '''
        Load the model from a pickle file
        :param file_path: File path to the pickle file with the model
        :return:
        '''
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print('done')


class Basic_ML(ML_models):
    def __init__(self, num_of_labels, data_path=None):
        super().__init__(data_path=data_path, num_of_labels=num_of_labels)
        self.epochs = 50
        self.learning_rate = 0.0001

    def create_model(self):
        '''
        Create the model architecture for the basic NN
        :return:
        '''
        self.modeltype = 'Basic_ML'
        self.MFCC_train, self.MFCC_test, self.label_train, self.label_test = train_test_split(self.mfcc_input,
                                                                                              self.target_instrument,
                                                                                              test_size=self.test_size)
        # Dropout layers and Regularising to reduce overfitting
        model = keras.Sequential([

            keras.layers.Flatten(input_shape=(self.mfcc_input.shape[1], self.mfcc_input.shape[2])),

            keras.layers.Dense(512, input_dim=self.MFCC_train.shape[1], activation='relu',
                               kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(self.num_of_labels, activation=self.final_activation)
        ])
        self.model = model

        return True


class CNN(ML_models):
    def __init__(self, num_of_labels, data_path=None):
        super().__init__(data_path=data_path, num_of_labels=num_of_labels)
        self.X_validation = None
        self.y_validation = None
        self.epochs = 30
        self.learning_rate = 0.0001

    def prepare_data(self):
        '''
        Prepare the data for the CNN model by splitting data into train, test, validation, then adding an
        additional AXIS to the inputs to make it 3D
        :return:
        '''
        self.MFCC_train, self.MFCC_test, self.label_train, self.label_test = train_test_split(self.mfcc_input,
                                                                                              self.target_instrument,
                                                                                              test_size=self.test_size)
        self.MFCC_train, self.X_validation, self.label_train, self.y_validation = train_test_split(self.MFCC_train,
                                                                                                   self.label_train,
                                                                                                   test_size=self.validation_size)

        # add an axis to input sets
        self.MFCC_train = self.MFCC_train[..., np.newaxis]
        self.X_validation = self.X_validation[..., np.newaxis]
        self.MFCC_test = self.MFCC_test[..., np.newaxis]

    def create_model(self):
        '''
        Create the model architecture for the CNN
        :return:
        '''
        self.modeltype = 'CNN'
        input_shape = (self.MFCC_train.shape[1], self.MFCC_train.shape[2], 1)  # 130, 13

        # Build network topology
        model = keras.Sequential()

        # 1st conv layer
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        model.add(keras.layers.BatchNormalization())

        # 2nd conv layer
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((2, 2)))

        # 3rd conv layer
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((2, 2)))

        # 4th conv layer
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((2, 2)))

        # Flatten the output and feed it through the dense layer
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.5))

        # Output layer
        model.add(keras.layers.Dense(self.num_of_labels, activation=self.final_activation))

        self.model = model

    def predict_model(self, MFCC):
        '''
        Predict the model
        :param MFCC: MFCC input data
        :return:
        '''

        # Add a dimension to input data for sample
        MFCC = MFCC[np.newaxis, ...]

        # Perform prediction
        prediction = self.model.predict(MFCC)

        # Get the top three predictions
        top_three_indices = np.argsort(prediction)
        top_three_indices = (top_three_indices.tolist())[0]
        top_three_indices = top_three_indices[::-1][:3]

        return top_three_indices


class RNN(ML_models):
    def __init__(self, num_of_labels, data_path=None):
        super().__init__(data_path=data_path, num_of_labels=num_of_labels)
        self.epochs = 30
        self.learning_rate = 0.0001
        self.num_of_labels = num_of_labels

    def prepare_data(self):
        '''
        Split data into train, test and validation sets
        :return:
        '''
        self.MFCC_train, self.MFCC_test, self.label_train, self.label_test = train_test_split(self.mfcc_input,
                                                                                              self.target_instrument,
                                                                                              test_size=self.test_size)
        self.MFCC_train, self.X_validation, self.label_train, self.y_validation = train_test_split(self.MFCC_train,
                                                                                                   self.label_train,
                                                                                                   test_size=self.validation_size)

    def create_model(self):
        '''
        Design the model architecture for an RNN
        :return:
        '''
        self.modeltype = 'RNN'
        input_shape = (self.MFCC_train.shape[1], self.MFCC_train.shape[2])

        model = keras.Sequential()

        # 3 LSTM layers
        model.add(keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.LSTM(64, return_sequences=True))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.LSTM(32))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())

        # Dense layer
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())

        # Output layer
        model.add(keras.layers.Dense(self.num_of_labels, activation=self.final_activation))

        self.model = model
