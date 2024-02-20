import pygame
import sys

# Define instrument names
instruments = {
    '0': 'accordion', '1': 'banjo', '2': 'bass', '3': 'cello', '4': 'clarinet', '5': 'cymbals', '6': 'drums',
    '7': 'flute', '8': 'guitar', '9': 'mallet_percussion', '10': 'mandolin', '11': 'organ', '12': 'piano',
    '13': 'saxophone', '14': 'synthesizer', '15': 'trombone', '16': 'trumpet', '17': 'ukulele',
    '18': 'violin', '19': 'voice'
}


# Initialize Pygame
def run(audio):
    '''
    Display the instrument predictions
    :param audio:
    :return:
    '''
    pygame.init()

    # Set up the window
    width = 800
    height = 600
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Instrument Display')

    # Set up colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (50, 50, 50)
    GREEN = (27, 242, 34)
    ORANGE = (255, 153, 0)
    RED = (255, 0, 0)

    # Load font
    instrument_font = pygame.font.Font(r'E:\NEA_Files\Eazy Chat.ttf', 15)
    instrument_values = list(instruments.values())

    # Calculate dimensions for each instrument cell
    num_columns = 5
    num_rows = (len(instrument_values) + num_columns - 1) // num_columns
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
        for i, instrument in enumerate(instrument_values):
            row = i // num_columns
            col = i % num_columns

            # Calculate positions for label and rectangle
            label_rect = pygame.Rect(col * cell_width, row * cell_height, cell_width, cell_height * 2 // 3)
            rect_rect = pygame.Rect(col * cell_width, row * cell_height + cell_height * 2 // 3, cell_width,
                                    cell_height // 3)

            # Render label with border
            text_surface = instrument_font.render(instrument, True, WHITE)
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
