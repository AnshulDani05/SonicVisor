# self.modeltype = 'CNN'
#         input_shape = (self.MFCC_train.shape[1], self.MFCC_train.shape[2], 1)  # 130, 13
#
#         # Build network topology
#         model = keras.Sequential()
#
#         # 1st conv layer
#         model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
#         model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
#         model.add(keras.layers.BatchNormalization())
#
#         # 2nd conv layer
#         model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
#         model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
#         model.add(keras.layers.BatchNormalization())
#
#         # 3rd conv layer
#         model.add(keras.layers.Conv2D(256, (2, 2), activation='relu'))
#         model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
#         model.add(keras.layers.BatchNormalization())
#
#         # Flatten the output and feed it through the dense layer
#         model.add(keras.layers.Flatten())
#         model.add(keras.layers.Dense(64, activation='relu'))
#         model.add(keras.layers.Dropout(0.3))
#
#         # output layer
#         model.add(keras.layers.Dense(self.num_of_labels, activation=self.final_activation))
#
#         self.model = model

# build network topology
        # model = keras.Sequential()
        #
        # # 2 LSTM layers
        # model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
        # model.add(keras.layers.LSTM(64))
        #
        # # dense layer
        # model.add(keras.layers.Dense(64, activation='relu'))
        # model.add(keras.layers.Dropout(0.3))