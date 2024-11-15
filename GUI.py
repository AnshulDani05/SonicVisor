import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, \
    QWidget, QSlider
from PyQt5.QtGui import QPixmap  # PyQT is a GUI module ideal for desktop applications
from PyQt5.QtCore import Qt
import pygame  # Pygame is a GUI module ideal for video games / graphic displays
from pygame import mixer  # Allows audio playback within the Pygame framework
import numpy as np  # Library for numerical computing
from DisplayWindows import Genre_display, Instrument_display

from ProcessData import prediction_data  # Custom module to break down audio input into format for neural network
from ModelHandling import load_genre_NNs, load_instrument_NNs  # Custom module to load genre neural network models  # Custom module to load instrument neural network models
import os  # Module for interacting with file paths and directories
import AUDIO_VIS  # Custom module to display audio visualiser


# Dictionary mapping genre labels to genres
genres = {'0': 'blues', '1': 'classical', '2': 'country', '3': 'disco', '4': 'hiphop', '5': 'jazz', '6': 'metal',
          '7': 'pop', '8': 'reggae', '9': 'rock'}
# Dictionary mapping instrument labels to instruments
instruments = {'0': 'accordion', '1': 'banjo', '2': 'bass', '3': 'cello', '4': 'clarinet', '5': 'cymbals', '6': 'drums',
               '7': 'flute', '8': 'guitar', '9': 'mallet_percussion', '10': 'mandolin', '11': 'organ', '12': 'piano',
               '13': 'saxophone', '14': 'synthesizer', '15': 'trombone', '16': 'trumpet', '17': 'ukulele',
               '18': 'violin', '19': 'voice'}


class MusicPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.song_path = None
        self.song_list = None
        self.forwards = None  # Keeps track of how many forwards
        self.rewinds = None  # Keep track of how many rewinds
        self.initUI()  # Initialises the UI
        self.next_track_clicked = False
        self.previous_track_clicked = False

    def initUI(self):
        '''
        Initialise/Draw the UI
        :return:
        '''
        self.setWindowTitle('Audio Tools')
        self.setGeometry(100, 100, 800, 400)
        self.song_label = QLabel('Select a song', self)
        self.song_label.setStyleSheet('color:white')

        # Centralized layout
        central_widget = QWidget(self)
        central_layout = QHBoxLayout(central_widget)

        # Left side layout
        left_layout = QVBoxLayout()

        # Image box
        self.image_label = QLabel(self)
        self.image_label.setText('AUDIO PLAYER AND ANALYSER')
        self.image_label.setStyleSheet('color:white; font-size: 28pt')
        self.image_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.image_label, alignment=Qt.AlignTop | Qt.AlignCenter)
        left_layout.addWidget(self.song_label, alignment=Qt.AlignTop | Qt.AlignCenter)

        self.other_audio_files = QLabel(self)
        self.other_audio_files.setTextInteractionFlags(Qt.TextSelectableByMouse)
        left_layout.addWidget(self.other_audio_files)

        # Play, stop, select, rewind, forward, next, previous, and pause buttons
        self.play_button = QPushButton('Play', self)
        self.play_button.setStyleSheet('background-color: rgb(88,88,88); color:white')
        self.stop_button = QPushButton('Stop', self)
        self.stop_button.setStyleSheet('background-color: rgb(88,88,88); color:white')
        self.select_button = QPushButton('Select Song', self)
        self.select_button.setStyleSheet('background-color: rgb(88,88,88); color:white')
        self.rewind_button = QPushButton('Rewind -10s', self)
        self.rewind_button.setStyleSheet('background-color: rgb(88,88,88); color:white')
        self.forward_button = QPushButton('Forward +10s', self)
        self.forward_button.setStyleSheet('background-color: rgb(88,88,88); color:white')
        self.next_button = QPushButton('Next Track', self)
        self.next_button.setStyleSheet('background-color: rgb(88,88,88); color:white')
        self.previous_button = QPushButton('Previous Track', self)
        self.previous_button.setStyleSheet('background-color: rgb(88,88,88); color:white')
        self.pause_button = QPushButton('Pause', self)
        self.pause_button.setStyleSheet('background-color: rgb(88,88,88); color:white')

        # Add the buttons to the left-hand display
        left_layout.addWidget(self.play_button)
        left_layout.addWidget(self.stop_button)
        left_layout.addWidget(self.select_button)
        left_layout.addWidget(self.rewind_button)
        left_layout.addWidget(self.forward_button)
        left_layout.addWidget(self.next_button)
        left_layout.addWidget(self.previous_button)
        left_layout.addWidget(self.pause_button)

        # Connect the buttons with their functions
        self.play_button.clicked.connect(self.play_music)
        self.stop_button.clicked.connect(self.stop_music)
        self.select_button.clicked.connect(self.choose_song)
        self.rewind_button.clicked.connect(self.rewind_music)
        self.forward_button.clicked.connect(self.forward_music)
        self.next_button.clicked.connect(self.next_track)
        self.previous_button.clicked.connect(self.previous_track)
        self.pause_button.clicked.connect(self.pause_music)

        # Volume slider
        self.volume_slider = QSlider(Qt.Horizontal, self)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)
        self.volume_slider.setToolTip("Volume")
        self.volume_slider.valueChanged.connect(self.change_volume)

        left_layout.addWidget(self.volume_slider)

        left_layout.addStretch()

        central_layout.addLayout(left_layout)

        # Right side layout
        right_layout = QVBoxLayout()
        self.paused = False

        # Instruments, Genre and Audio Visualiser buttons with images
        self.instruments_button = QPushButton('Instruments', self)
        self.instruments_button.setStyleSheet('background-color: rgb(88,88,88); color:white')
        self.instruments_image = QLabel(self)
        self.instruments_image.setPixmap(QPixmap(r'E:\NEA_Files\Instruments.png').scaled(100, 100,
                                                                                         Qt.KeepAspectRatio))  # Replace 'path_to_image_placeholder.png' with the path to your image placeholder
        self.instruments_image.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.instruments_button)
        right_layout.addWidget(self.instruments_image)
        self.instruments_button.clicked.connect(self.instrument_identifier)

        self.genre_button = QPushButton('Genre', self)
        self.genre_button.setStyleSheet('background-color: rgb(88,88,88); color:white')
        self.genre_image = QLabel(self)
        self.genre_image.setPixmap(QPixmap(r'E:\NEA_Files\Genres.jpg').scaled(100, 100, Qt.KeepAspectRatio))
        self.genre_image.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.genre_button)
        right_layout.addWidget(self.genre_image)
        self.genre_button.clicked.connect(self.genre_identifier)

        self.audio_visualiser = QPushButton('Audio Visualiser', self)
        self.audio_visualiser.setStyleSheet('background-color: rgb(88,88,88); color:white')
        self.audio_visualiser_image = QLabel(self)
        self.audio_visualiser_image.setPixmap(QPixmap(r'E:\NEA_Files\WAVE.png').scaled(200, 200, Qt.KeepAspectRatio))
        self.audio_visualiser_image.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.audio_visualiser)
        right_layout.addWidget(self.audio_visualiser_image)
        self.audio_visualiser.clicked.connect(self.audio_vis)

        central_layout.addLayout(right_layout)

        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

    def choose_song(self):
        '''
        Choose a song from the user's file library, and check the folder for other sound files
        :return:
        '''
        mixer.init()
        self.song_path, _ = QFileDialog.getOpenFileName(self, 'Open Music File', '', 'Audio Files (*.mp3 *.ogg *.wav)')
        if self.song_path:
            self.song_label.setText(f"Selected: {os.path.basename(self.song_path)}")
            self.update_song_list()
            self.play_music(new_song=True)

    def update_song_list(self):
        '''
        Check the current song path directory for other sound files
        :return:
        '''
        if self.song_path:
            folder_path = os.path.dirname(self.song_path)
            self.song_list = [file for file in os.listdir(folder_path) if
                              file.lower().endswith(('.mp3', '.ogg', '.wav'))]
            print(self.song_list)
            current_index = self.song_list.index(os.path.basename(self.song_path))
            self.other_song_list = self.song_list[current_index:] + self.song_list[:current_index]
            if len(self.other_song_list) > 1:
                self.other_song_list = "\n".join(self.other_song_list)
                self.other_audio_files.setText(f'Other audio files: \n{self.other_song_list}')
            else:
                self.other_audio_files.setText('No other audio files in the folder')

    def play_music(self, new_song=False):
        '''
        Play the music. Check if the new track is due to the next or previous track it being a new song.
        If so, it will reset the counters i.e forwards, to 0
        :param new_song: check if it is a new song
        :return:
        '''
        if self.song_path:
            if new_song or self.next_track_clicked or self.previous_track_clicked:
                self.next_track_clicked = False
                self.previous_track_clicked = False
                self.forwards = 0
                self.rewinds = 0
                mixer.music.load(self.song_path)
                mixer.music.play()
            elif pygame.mixer.music.get_busy() and pygame.mixer.music.get_pos() > 0 or self.paused:
                # Get the current playback position before pausing
                current_pos = pygame.mixer.music.get_pos() // 1000
                current_pos += (self.forwards * 10)
                current_pos -= (self.rewinds * 10)
                pygame.mixer.music.stop()
                mixer.music.load(self.song_path)
                # Set the position and resume playback
                pygame.mixer.music.play(start=current_pos, fade_ms=-1)
                if self.paused:
                    self.paused = False
            else:
                # Start playing from the beginning if it's a new song or after reaching the end
                self.forwards = 0
                self.rewinds = 0
                mixer.music.load(self.song_path)
                mixer.music.play()

    def pause_music(self):
        '''
        Pause the music
        :return:
        '''
        if mixer.music.get_busy():
            pygame.mixer.music.pause()
            self.paused = True
            print('Paused')

    def stop_music(self):
        '''
        Stop the music
        :return:
        '''
        mixer.music.stop()

    def rewind_music(self):
        '''
        Rewind the music by 10 seconds
        :return:
        '''
        if self.song_path:
            # Calculate the new position of the audio
            current_pos = pygame.mixer.music.get_pos() // 1000  # convert to seconds
            current_pos = current_pos - (10 * self.rewinds)
            current_pos = current_pos + (10 * self.forwards)
            self.rewinds += 1
            new_pos = max(current_pos - 10, 0)
            # Set the new position and resume playback
            pygame.mixer.music.play(start=new_pos)

    def forward_music(self):
        '''
        Forward the music by 10 seconds
        :return:
        '''
        if self.song_path and pygame.mixer.music.get_busy():
            # Pause the music as you are skipping to a new position
            pygame.mixer.music.pause()

            # Calculate the new position
            current_pos = pygame.mixer.music.get_pos() // 1000  # convert to seconds
            current_pos = current_pos + (10 * self.forwards)
            current_pos = current_pos - (10 * self.rewinds)
            self.forwards = self.forwards + 1
            new_pos = current_pos + 10

            # Set the new position and resume playback
            pygame.mixer.music.play(start=new_pos)

    def next_track(self):
        '''
        Skip to the next track
        :return:
        '''
        if self.song_list:
            if self.song_path:
                # Skip to the next song
                current_index = self.song_list.index(os.path.basename(self.song_path))
                next_index = (current_index + 1) % len(self.song_list)
                self.song_path = os.path.join(os.path.dirname(self.song_path), self.song_list[next_index])
                self.next_track_clicked = True
            else:
                # If no current song is selected, start from the first one in the list
                self.song_path = os.path.join(os.path.dirname(self.song_path), self.song_list[0])

            # Play the music and set the new name
            self.play_music()
            self.song_label.setText(f"Selected: {os.path.basename(self.song_path)}")

    def previous_track(self):
        '''
        Go to the previous track
        :return:
        '''
        if self.song_list:
            if self.song_path:
                # Go to the previous track in the song list
                current_index = self.song_list.index(os.path.basename(self.song_path))
                previous_index = (current_index - 1) % len(self.song_list)
                self.song_path = os.path.join(os.path.dirname(self.song_path), self.song_list[previous_index])
                self.previous_track_clicked = True
            else:
                # If no current song is selected, start from the last one in the list
                self.song_path = os.path.join(os.path.dirname(self.song_path), self.song_list[-1])

            self.play_music()
            self.song_label.setText(f"Selected: {os.path.basename(self.song_path)}")

    def genre_identifier(self):
        '''
        Process the current audio file, run it through the neural network to predict genres, then display the
        predictions
        :return:
        '''
        if self.song_path:
            # Split the audio file into frames, and run each of them through the neural network predictor
            # Get the top three predictions
            indexes = [0 for i in range(10)]
            Model = load_genre_NNs.Load_CNN(10)
            mfccs = prediction_data.process_data(self.song_path)
            count = 0
            for mfcc in mfccs:
                count += 1
                predictions = Model.predict_model(mfcc)
                indexes[predictions[0]] = indexes[predictions[0]] + 3
                indexes[predictions[1]] = indexes[predictions[1]] + 2
                indexes[predictions[2]] = indexes[predictions[2]] + 1

            indexes = np.argsort(indexes)
            indexes = indexes.tolist()
            indexes = indexes[::-1][:3]
            print(indexes)
            current_pos = pygame.mixer.music.get_pos() // 1000
            self.hide()
            Genre_display.run(indexes)
            self.show()
            mixer.init()
            mixer.music.load(self.song_path)
            pygame.mixer.music.play(start=current_pos, fade_ms=-1)
            print(self.song_path)

    def instrument_identifier(self):
        '''
        Process the current audio file, run it through the neural network to predict instruments,
        then display the predictions
        :return:
        '''
        if self.song_path:
            # Split the audio file into frames, and run each of them through the neural network predictor
            # Get the top three predictions
            indexes = [0 for i in range(20)]
            Model = load_instrument_NNs.Load_CNN(20)
            mfccs = prediction_data.process_data(self.song_path)
            for mfcc in mfccs:
                predictions = Model.predict_model(mfcc)
                indexes[predictions[0]] = indexes[predictions[0]] + 3
                indexes[predictions[1]] = indexes[predictions[1]] + 2
                indexes[predictions[2]] = indexes[predictions[2]] + 1

            indexes = np.argsort(indexes)
            indexes = indexes.tolist()
            indexes = indexes[::-1][:3]
            print(indexes)
            current_pos = pygame.mixer.music.get_pos() // 1000
            self.hide()
            Instrument_display.run(indexes)
            self.show()
            mixer.init()
            mixer.music.load(self.song_path)
            pygame.mixer.music.play(start=current_pos, fade_ms=-1)
            print(self.song_path)

    def audio_vis(self):
        '''
        Display the audio visualiser
        :return:
        '''
        audio_visualiser = AUDIO_VIS.AudioVisualiser(self.song_path)
        pygame.mixer.music.stop()
        self.hide()
        audio_visualiser.draw_image()
        self.show()
        mixer.init()

    def change_volume(self, value):
        '''
        Change the volume
        :param value: the value to make the volume
        :return:
        '''
        if self.song_path:
            volume = value / 100.0
            pygame.mixer.music.set_volume(volume)

    def closeEvent(self, event):
        '''
        Close the player and stop the music when the X is clicked
        :param event:
        :return:
        '''
        mixer.quit()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = MusicPlayer()
    # Set background color
    player.setStyleSheet("background-color: rgb(50, 50, 50);")
    player.show()
    sys.exit(app.exec_())
