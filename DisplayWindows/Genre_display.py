import pygame
import sys

# Define instrument names
genres = {'0': 'blues', '1': 'classical', '2': 'country', '3': 'disco', '4': 'hiphop', '5': 'jazz', '6': 'metal',
          '7': 'pop', '8': 'reggae', '9': 'rock'}


# Initialize Pygame
def run(audio):
    '''
    Draw display
    :param audio: the indexes of the top 3 instruments recognised
    :return:
    '''

    pygame.init()

    # Set up the window
    width = 800
    height = 600
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Genre Display')

    # Set up colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (50, 50, 50)
    GREEN = (27, 242, 34)
    ORANGE = (255, 153, 0)
    RED = (255, 0, 0)

    # Load font
    genre_font = pygame.font.Font(r'E:\NEA_Files\Eazy Chat.ttf', 30)
    genre_values = list(genres.values())

    # Calculate dimensions for each instrument cell
    num_columns = 5
    num_rows = (len(genre_values) + num_columns - 1) // num_columns
    cell_width = width // num_columns
    cell_height = height // num_rows

    # Main loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True

        # Fill the background with white
        window.fill(GRAY)

        # Render instrument labels and colored rectangles
        for i, instrument in enumerate(genre_values):
            row = i // num_columns
            col = i % num_columns

            # Calculate positions for label and rectangle
            label_rect = pygame.Rect(col * cell_width, row * cell_height, cell_width, cell_height * 2 // 3)
            rect_rect = pygame.Rect(col * cell_width, row * cell_height + cell_height * 2 // 3, cell_width,
                                    cell_height // 3)

            # Render label with border
            text_surface = genre_font.render(instrument, True, WHITE)
            text_rect = text_surface.get_rect(center=label_rect.center)
            window.blit(text_surface, text_rect)

            # Render colored rectangle with border
            pygame.draw.rect(window, BLACK, rect_rect, 2)
            color = GRAY
            if i in audio:
                if audio.index(i) == 0:
                    color = GREEN
                elif audio.index(i) == 1:
                    color = ORANGE
                elif audio.index(i) == 2:
                    color = RED

            pygame.draw.rect(window, color, rect_rect.inflate(-4, -4))  # Inner rectangle

        # Update the display
        pygame.display.update()

    # Quit Pygame
    pygame.quit()
    sys.exit()
