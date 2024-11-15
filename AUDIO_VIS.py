# This code produces a graphic display which reacts to the music being played
# It visualizes an audio file by displaying bars and splines that react to the music being played

import os.path  # Module for interacting with file paths
import pygame  # Library for creating a GUI
import random  # Module for generating random numbers
import numpy as np  # Library for numerical computing
import colorsys  # Module for converting between RGB and HSV colour spaces
import librosa  # Library for audio and music analysis

N_FFT = 2048  # Number of points used in the FFT
SR = 22050  # Sample Rate


class AudioVisualiser:
    def __init__(self, audio_file):
        '''
        Initalises all of the constants/variables that will be used to produce the graphic image
        :param audio_file: Path to the audio file
        '''
        self.splines = True
        self.bars = True
        self.width = 800
        self.height = 600
        self.audio = audio_file  # the audio file
        self.fps = 30  # the rate at which the pygame window will update
        self.max_bar_height = 100  # the maximum height of the bar in the visualisation
        self.original_centre_circle_r = 50  # original radius of the centre circle
        self.centre_circle_r = 50  # the current radius of the centre circle
        self.bar_scale_factor = 500  # the factor to scale the bars in the simulation
        self.rotate_angle = 0  # angle by which the visualisation should rotate
        self.window_size = 1 / self.fps  # size of the window used for analysis within the stft process
        self.colour_change_interval = 1  # interval at which the colour change in multi-coloured setting should happen
        self.last_colour_change_time = pygame.time.get_ticks() / 1000.0  # time of the last colour change
        self.hue = 0.0  # hue value for colour generation
        self.current_hue = random.uniform(0, 1)  # current hue value
        self.multi_coloured = False  # boolean value indicating whether multi-coloured mode is on
        self.colour = (255, 255, 255)  # current colour for single colour setting
        self.colour_states = {
            'red': False,
            'blue': False,
            'green': False,
            'purple': False,
            'brown': False,
            'multi_coloured': False
        }  # states of different colours for visualisation

    def colour_picker(self):
        '''
        Decide the colour
        :return:
        '''
        if self.multi_coloured:
            current_time = pygame.time.get_ticks() / 1000.0
            if current_time - self.last_colour_change_time >= self.colour_change_interval:
                # Generate a random hue for next colour scheme
                self.current_hue = random.uniform(0, 1)
                self.last_colour_change_time = current_time
                r, g, b = colorsys.hsv_to_rgb(self.current_hue, 1.0, 1.0)
                self.colour = (int(r * 255), int(g * 255), int(b * 255))
        else:
            # Check if any of the colour states are activated in the single-colour mode
            # If not it will default to white
            for colour in self.colour_states:
                if self.colour_states[colour]:
                    if colour == 'red':
                        self.colour = (255, 0, 0)
                    elif colour == 'blue':
                        self.colour = (0, 0, 255)
                    elif colour == 'green':
                        self.colour = (0, 255, 0)
                    elif colour == 'purple':
                        self.colour = (128, 0, 128)
                    elif colour == 'brown':
                        self.colour = (165, 42, 42)
                    break
            else:
                self.colour = (255, 255, 255)

    def draw_bars(self, wave):
        '''
        Draw the bars and circles for the input wave frame. Checks the colour settings, and if multi-coloured
        is selected, and sufficient time has passed, it will randomly generate a new hue.
        :param wave: the audio wave data
        :return:
        '''

        for i, plot in enumerate(wave):
            # For each point in the wave calculate where to plot the lines/bars and the circle at the end and draw it
            angle = i * self.angle_delta - self.rotate_angle
            start_point = (self.centre_circle_r * np.cos(angle) + self.width / 2,
                           self.centre_circle_r * np.sin(angle) + self.height / 2)
            coord = self.translate((self.bar_scale_factor * plot + self.centre_circle_r) * np.cos(angle),
                                   (self.bar_scale_factor * plot + self.centre_circle_r) * np.sin(angle))
            pygame.draw.circle(self.screen, self.colour, coord, 3)
            pygame.draw.line(self.screen, self.colour, start_point, coord, 2)

    def draw_splines(self, wave, n_lines=1):
        '''
        Draw splines for the input wave.
        :param wave: the audio wave data
        :param n_lines: the number of splines to draw
        :return:
        '''

        spline_points = []
        factor = self.bar_scale_factor

        for n_factor in range(n_lines):
            factor = factor / (n_factor + 1)
            # Iterate through each point in the wave
            for i, plot in enumerate(wave):
                # Calculate the angle from the center and new coordinates
                angle = i * self.angle_delta - self.rotate_angle
                coord = self.translate((factor * plot + self.centre_circle_r) * np.cos(angle),
                                       (factor * plot + self.centre_circle_r) * np.sin(angle))
                spline_points.append(coord)

            # Draw the spline curve using the point
            for i in range(len(spline_points) - 1):
                p0, p1, p2, p3 = spline_points[max(
                    0, i - 1)], spline_points[i], spline_points[i + 1], spline_points[
                                     min(len(spline_points) - 1, i + 2)]
                for t in range(15):
                    t /= 10.0
                    # Calculate intermediate points using cubic Bezier curve formula
                    x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 *
                                                                     p2[0] - p3[0]) * t ** 2 + (
                                       -p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t ** 3)
                    y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 *
                                                                     p2[1] - p3[1]) * t ** 2 + (
                                       -p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t ** 3)
                    pygame.draw.line(self.screen, self.colour, (x, y), (x, y), 2)

    def translate(self, x, y):
        '''
        Translate the given points from a normalized space to screen coordinates
        :param x: The x-coordinate value
        :param y: The y-coordinate value
        :return: Translated screen coordinates (screen_x, screen_y)
        '''
        # Calculate scaling factors to map the normalized space to the screen size
        x_scale_factor = self.width / (2 * (self.max_bar_height + self.centre_circle_r))
        y_scale_factor = self.width / (2 * (self.max_bar_height + self.centre_circle_r))

        # Calculate the center of the screen
        screen_center_x = self.width / 2
        screen_center_y = self.height / 2

        # Translate the normalized coordinates to screen coordinates
        screen_x = x_scale_factor * x + screen_center_x
        screen_y = y_scale_factor * y + screen_center_y

        return screen_x, screen_y

    def get_stft(self):
        '''
        Get the STFTs of the Audio data and normalise them in the range (0, 1)
        Also get the tempo of the audio data, so the multi-coloured setting can be synced with the audio
        '''
        audio, sr = librosa.load(self.audio, sr=SR)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=SR)
        HOP_LENGTH = int(self.window_size * SR)
        frequency_amplitude = np.abs(librosa.stft(audio, n_fft=2048, hop_length=HOP_LENGTH))
        max_val = np.max(frequency_amplitude)  # Get the max value in the wave, so all the values can be
        # squashed into the range (0-1)
        for i in range(np.shape(frequency_amplitude)[0]):
            for j in range(np.shape(frequency_amplitude)[1]):
                frequency_amplitude[i][j] = (frequency_amplitude[i][j]) / max_val

        frequency_amplitude = frequency_amplitude.T
        return frequency_amplitude, tempo

    def shorten_wave(self, wave, n_nodes):
        '''
        Shortens the wave to size n_nodes, by figuring out the interval step, and only taking the value
        from the new index values
        :param wave: input wave
        :param n_nodes: number of values to shorten it to
        :return:
        '''
        interval = len(wave) // n_nodes
        new_wave = wave[::interval]
        return new_wave

    def draw_image(self):
        '''
        Draw the whole audio visualiser display. Utilises all the previously defined methods.
        :return:
        '''
        fps = 30
        data, tempo = self.get_stft()  # Get the data and the tempo

        self.colour_change_interval = 60 / tempo

        pygame.init()
        clock = pygame.time.Clock()

        title_font = pygame.font.Font(r'E:\NEA_Files\security-meltdown.regular.ttf', 50)
        font = pygame.font.Font(r'E:\NEA_Files\security-meltdown.regular.ttf', 25)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Audio Visualiser")

        # Specify the colour rectangles location and size
        checkbox_size = 20
        checkbox_margin_horizontal = 20
        checkbox_margin_vertical = 10
        checkbox_start_y = self.height - 6 * (checkbox_size + checkbox_margin_vertical) - 10
        checkbox_rects = {
            'red': pygame.Rect(self.width - checkbox_size - checkbox_margin_horizontal * 10, checkbox_start_y,
                               checkbox_size, checkbox_size),
            'blue': pygame.Rect(self.width - checkbox_size - checkbox_margin_horizontal * 10,
                                checkbox_start_y + (checkbox_size + checkbox_margin_vertical) * 1, checkbox_size,
                                checkbox_size),
            'green': pygame.Rect(self.width - checkbox_size - checkbox_margin_horizontal * 10,
                                 checkbox_start_y + (checkbox_size + checkbox_margin_vertical) * 2, checkbox_size,
                                 checkbox_size),
            'purple': pygame.Rect(self.width - checkbox_size - checkbox_margin_horizontal * 10,
                                  checkbox_start_y + (checkbox_size + checkbox_margin_vertical) * 3, checkbox_size,
                                  checkbox_size),
            'brown': pygame.Rect(self.width - checkbox_size - checkbox_margin_horizontal * 10,
                                 checkbox_start_y + (checkbox_size + checkbox_margin_vertical) * 4, checkbox_size,
                                 checkbox_size),
            'multi_coloured': pygame.Rect(self.width - checkbox_size - checkbox_margin_horizontal * 10,
                                          checkbox_start_y + (checkbox_size + checkbox_margin_vertical) * 5,
                                          checkbox_size, checkbox_size),
            'splines': pygame.Rect(self.width - checkbox_size - checkbox_margin_horizontal * 10,
                                   10,
                                   checkbox_size, checkbox_size),
            'bars': pygame.Rect(self.width - checkbox_size - checkbox_margin_horizontal * 10,
                                10 + checkbox_size + checkbox_margin_vertical,
                                checkbox_size, checkbox_size)
        }

        pygame.mixer.init()
        pygame.mixer.music.load(self.audio)  # Load the audio
        pygame.mixer.music.play()  # Play the music

        running = True

        # Main pygame loop
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        clicked_checkbox = None
                        for colour, rect in checkbox_rects.items():
                            if rect.collidepoint(event.pos):
                                # Identify the clicked checkbox
                                clicked_checkbox = colour
                                break

                        if clicked_checkbox == 'splines':
                            self.splines = not self.splines
                        elif clicked_checkbox == 'bars':
                            self.bars = not self.bars
                        elif clicked_checkbox == 'multi_coloured':
                            # Toggle multi-colored mode
                            self.multi_coloured = not self.multi_coloured
                            if self.multi_coloured:
                                # If multi-colored mode is activated, deactivate all color states
                                for key in self.colour_states:
                                    self.colour_states[key] = False
                        else:
                            # Handle color checkboxes
                            if clicked_checkbox:
                                # Activate only the clicked color checkbox, deactivate others
                                self.multi_coloured = False
                                for key in self.colour_states:
                                    self.colour_states[key] = key == clicked_checkbox


            elapsed_time = pygame.mixer.music.get_pos() / 1000.0
            # Make the current data_index the position in the data that corresponds to the current wave
            # This stage makes time delays redundant, ensuring the current display corresponds to the current audio
            data_indx = int(elapsed_time / self.window_size) % len(data)

            wave = self.shorten_wave(data[data_indx], 75)
            n_bars = len(wave)
            self.angle_delta = 2 * np.pi / n_bars  # The angle separation between each bar
            wave_max = max(wave)
            # Change the radius of the centre circle depending on the max amplitude from the wave
            self.centre_circle_r = self.original_centre_circle_r + 25 * wave_max

            # Fill the screen background then draw the bars and splines
            self.screen.fill((50, 50, 50))
            self.colour_picker()
            if self.bars:
                self.draw_bars(wave)
            if self.splines:
                self.draw_splines(wave, 3)

            self.rotate_angle += np.radians(1)

            # Draw checkboxes
            for colour, rect in checkbox_rects.items():
                pygame.draw.rect(self.screen, (255, 255, 255), rect)
                if colour in self.colour_states:
                    if self.colour_states[colour] or (colour == 'multi_coloured' and self.multi_coloured):
                        pygame.draw.line(self.screen, (0, 0, 0), (rect.x + 5, rect.y + 5),
                                         (rect.x + rect.width - 5, rect.y + rect.height - 5), 2)
                        pygame.draw.line(self.screen, (0, 0, 0), (rect.x + 5, rect.y + rect.height - 5),
                                         (rect.x + rect.width - 5, rect.y + 5), 2)
                else:
                    if (colour == 'bars' and self.bars) or (colour == 'splines' and self.splines):
                        pygame.draw.line(self.screen, (0, 0, 0), (rect.x + 5, rect.y + 5),
                                         (rect.x + rect.width - 5, rect.y + rect.height - 5), 2)
                        pygame.draw.line(self.screen, (0, 0, 0), (rect.x + 5, rect.y + rect.height - 5),
                                         (rect.x + rect.width - 5, rect.y + 5), 2)

            # Draw colour labels
            labels = ['red', 'blue', 'green', 'purple', 'brown', 'multi-coloured']
            for i, label in enumerate(labels):
                text_surface = font.render(label, True, pygame.Color('green'))
                self.screen.blit(text_surface, dest=(self.width + checkbox_size - checkbox_margin_horizontal * 10,
                                                     checkbox_start_y + i * (checkbox_size + checkbox_margin_vertical)))

            text_surface = font.render('splines', True, pygame.Color('green'))
            self.screen.blit(text_surface, dest=(self.width + checkbox_size - checkbox_margin_horizontal * 10,
                                                 10))

            text_surface = font.render('bars', True, pygame.Color('green'))
            self.screen.blit(text_surface, dest=(self.width + checkbox_size - checkbox_margin_horizontal * 10,
                                                 10 + checkbox_size + checkbox_margin_vertical))

            # Draw the text
            text_surface = title_font.render('AUDIO VISUALISER:', True, pygame.Color('green'))
            self.screen.blit(text_surface, dest=(10, 10))

            text_surface = font.render(str(os.path.basename(self.audio)), True, pygame.Color('green'))
            self.screen.blit(text_surface, dest=(10, 550))

            pygame.display.flip()
            clock.tick(fps)
