"""
This file contains the main script and graphical engine of the game. Run this script in order to
run the game.
"""
from re import compile

import pygame
import pygame as p
import os


from engine import *
from agents import *
from constants import SQ_SIZE, BOARD_HEIGHT, BOARD_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH, MAX_FPS, RECOMMENDER_TYPE

# Change directory
os.chdir("../")

# Initialize graphical engine
p.init()
TEXT_FONT = p.font.Font(None, 32)
IMAGES = {}

# Set screen and clock
screen = p.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
screen.fill(p.Color("white"))
clock = p.time.Clock()

# Initialize Game state and agents
gs = GameState()
recommender = Recommender(gs, version=RECOMMENDER_TYPE)
w_agent = LogicalAgent()        # Werewolf agent
d_agent = LogicalAgent()        # Demon Portal agent
v_agent = LogicalAgent()        # Village agent


def load_images():
    """
    Loads all images that are going to be used in the game
    """
    images = ['player', 'werewolf', 'demon_portal', 'village', 'circle', 'recommendation']
    for image in images:
        IMAGES[image] = p.transform.scale(p.image.load("./images/" + image + ".png"), (SQ_SIZE, SQ_SIZE))
    IMAGES['book'] = p.transform.scale(p.image.load("./images/book.png"), (SQ_SIZE * 2 / 3, SQ_SIZE * 2 / 3))
    IMAGES['close'] = p.transform.scale(p.image.load("./images/close.png"), (80, 80))
    IMAGES['map'] = p.transform.scale(p.image.load("./images/map.png"), (BOARD_HEIGHT * SQ_SIZE, BOARD_WIDTH * SQ_SIZE))


# Load images
load_images()  # This only applies once


class Button(object):
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.clicked = False

    def draw(self, surface):
        action = False
        # Get mouse position
        pos = p.mouse.get_pos()
        if self.rect.collidepoint(pos):
            if p.mouse.get_pressed()[0] == 1 and self.clicked == False:
                self.clicked = True
                action = True

        if p.mouse.get_pressed()[0] == 0:
            self.clicked = False

        surface.blit(self.image, (self.rect.x, self.rect.y))
        return action


# Create "book" menu
rect_width = int(SCREEN_WIDTH * 0.9)
balloon_image = p.image.load("./images/balloon.png")
rect_height = int(balloon_image.get_height() * rect_width / balloon_image.get_width())
IMAGES["balloon"] = p.transform.scale(balloon_image, (rect_width - 20, rect_height))

# Calculate the position of the rectangle
rect_x = (SCREEN_WIDTH - rect_width) // 2
rect_y = SCREEN_HEIGHT - 250

# Create and draw the rectangle
text_image = p.image.load("./images/text.png")
scaled_height = int(text_image.get_height() * (rect_width * 4/5) / text_image.get_width())
IMAGES["text"] = p.transform.scale(text_image, (rect_width * 4/5, scaled_height))

# Create "Diviner's book" button:
image_width, image_height = IMAGES["book"].get_size()
book = Button(SCREEN_WIDTH - image_width - 10, SCREEN_HEIGHT - image_height - 10, IMAGES["book"])


def draw_game_state(screen_, game_state):
    """
    Draws current state of the game (board and players, health)
    """
    screen_.blit(IMAGES["map"], p.Rect(0, 0, SCREEN_HEIGHT, SCREEN_WIDTH))
    shadow = p.Surface((SCREEN_HEIGHT, SCREEN_WIDTH))
    shadow.set_alpha(128)
    shadow.fill(p.color.Color('grey'))

    # Draw player
    x, y = gs.where_is_player()
    shadow.blit(IMAGES["circle"], (int(y) * SQ_SIZE, int(x) * SQ_SIZE))
    screen_.blit(IMAGES["player"], p.Rect(int(y) * SQ_SIZE, int(x) * SQ_SIZE, SQ_SIZE, SQ_SIZE))
    # Draw entities
    for x, y in entities_found:
        entity_type = None
        if game_state.board[x, y].startswith("D"):
            entity_type = "demon_portal"
        elif game_state.board[x, y].startswith("W"):
            entity_type = "werewolf"
        if entity_type is not None:
            shadow.blit(IMAGES["circle"], (int(y) * SQ_SIZE, int(x) * SQ_SIZE))
            screen_.blit(IMAGES[entity_type], p.Rect(int(y) * SQ_SIZE, int(x) * SQ_SIZE, SQ_SIZE, SQ_SIZE))
    # Draw recommendation
    if recommendation is not None:
        shadow.blit(IMAGES["circle"], (int(recommendation[1]) * SQ_SIZE, int(recommendation[0]) * SQ_SIZE))
        screen_.blit(IMAGES["recommendation"], p.Rect(int(recommendation[1]) * SQ_SIZE,
                                                      int(recommendation[0]) * SQ_SIZE,
                                                      SQ_SIZE, SQ_SIZE))

    screen_.blit(shadow, (0, 0), special_flags=p.BLEND_RGBA_SUB)

    # Draw player's health
    player_health = str(gs.entities["P1"].health)
    hp = TEXT_FONT.render(f"HP: {player_health}", True, "#ffffff")
    screen_.blit(hp, (SCREEN_WIDTH - 90, 20))

    # Draw player's senses
    color_smell = "#ffffff" if smell_active else "#7d7f7d"
    smell = TEXT_FONT.render(f"Smell", True, color_smell)
    screen_.blit(smell, (SCREEN_WIDTH - 90, 40))

    color_hear = "#ffffff" if hear_active else "#7d7f7d"
    hear = TEXT_FONT.render(f"Hear", True, color_hear)
    screen_.blit(hear, (SCREEN_WIDTH - 90, 60))

    color_see = "#ffffff" if see_active else "#7d7f7d"
    see = TEXT_FONT.render(f"See", True, color_see)
    screen_.blit(see, (SCREEN_WIDTH - 90, 80))
    return


input_rect = p.Rect(rect_x + 35, rect_y + 85, 200, 32)
valid_input_pattern = compile(r'^[DWV]\[\d+(,\d+)*]$')

act_color = p.Color('#ffa300')
pas_color = p.Color('#000000')


def open_book(screen_):
    """
    Draws the Diviners' Book menu.
    """
    global response_active, answer
    # Draw the balloon of the Diviners' Book
    screen_.blit(IMAGES["balloon"], p.Rect(rect_x, rect_y, 0, 0))
    screen_.blit(IMAGES["text"], p.Rect(rect_x + 20, rect_y + 30, 0, 0))

    # Draw the user's textbox
    screen_.blit(textbox, (rect_x + 40, rect_y + 90))
    if text_active:
        color = act_color
    else:
        color = pas_color
    p.draw.rect(screen_, color, input_rect, 2)

    # Draw the book's answer
    def is_valid_input(input_str):
        return bool(valid_input_pattern.match(input_str))

    if not text_active and response_active:
        if not is_valid_input(user_text):
            answer = TEXT_FONT.render("Not a valid input.", True, "#000000")
        else:
            searching = "Village"
            agent = v_agent
            if user_text.startswith("D"):
                searching = "Demon Portal"
                agent = d_agent
            elif user_text.startswith("W"):
                searching = "Werewolf"
                agent = w_agent
            response = agent.ask(user_text)

            answer_text = "There is a " + searching if response else "Probably no " + searching
            answer = TEXT_FONT.render(answer_text, True, "#000000")
    screen_.blit(answer, (rect_x + 320, rect_y + 90))
    response_active = False
    return


# Runs the program
if __name__ == "__main__":
    # Game global variables #
    running, game_paused, percept, text_active, response_active = True, False, True, False, False
    smell_active, hear_active, see_active = False, False, False
    new_state = 0, 0, False
    recommendation = None
    user_text = ''
    answer = ''
    entities_found = []

    # Prologue #
    print("Prologue:")
    print("You are sitting in a tavern. A picturesque man comes to you. He looks like a sage or cleric and he needs "
          "your help, the whole town needs it. A few days ago a group of people was completely massacred. One survivor "
          "said it was the fault of werewolves. Your task is to hunt down these werewolves. You carry an ax and a "
          "lucky charm whose eyes will shine pointing in the right direction. In addition, the cleric granted you a "
          "book enchanted by a Divine spell. This book possesses a power stronger than that of your amulet, "
          "since it reveals with total certainty if there is danger in the place that you wish to visit. If the book "
          "doesn't guarantee danger, then it's a matter of luck whether or not there is.")
    print()

    print("Game:")
    print("You come from a clan of warriors and hunters. You know the secrets of the forest, the monsters and the "
          "devilish. In the forest there are portals that lead to hell, guaranteeing the death of anyone who dares to "
          "go through them or even come close. It is a very dark night, and you should try to avoid these at all "
          "costs. You know that Demon portals have a characteristic smell, like rottenness. This will help you "
          "orient yourself. You will hear the howling of the werewolves if you are close to them, just as you will "
          "see the distant lights of the town if you approach it. Do not return to town even by mistake. You will "
          "seem like a coward, a person who is not capable of the task entrusted to you. If your weapon breaks then "
          "you will be vulnerable to the dangers that enter the forest, and you will surely die.")
    print()

    print("Controls:")
    print("WASD to move and ARROWS to attack")
    print()

    # Main game loop #
    # print(gs.board)       # Prints board with known locations.
    while running:
        # Draw current Game state
        if not game_paused:
            draw_game_state(screen, gs)
            if book.draw(screen):
                game_paused = True
                response_active = True
        else:
            if book.draw(screen):
                text_active = False
                game_paused = False
                user_text = ""
            textbox = TEXT_FONT.render(user_text, True, "#000000")
            input_rect.w = max(textbox.get_width() + 10, 200)
            open_book(screen)

        # Check events
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                # Book has been opened
                if game_paused and text_active:
                    # Delete key
                    if e.key == p.K_BACKSPACE:
                        user_text = user_text[0:-1]
                    # Add text
                    else:
                        user_text += e.unicode
                # In game
                else:
                    if e.key == p.K_w:
                        move = "UP"
                    elif e.key == p.K_UP:
                        move = "UP+ATTACK"
                    elif e.key == p.K_a:
                        move = "LEFT"
                    elif e.key == p.K_LEFT:
                        move = "LEFT+ATTACK"
                    elif e.key == p.K_s:
                        move = "DOWN"
                    elif e.key == p.K_DOWN:
                        move = "DOWN+ATTACK"
                    elif e.key == p.K_d:
                        move = "RIGHT"
                    elif e.key == p.K_RIGHT:
                        move = "RIGHT+ATTACK"
                    else:
                        continue
                    try:
                        new_state = gs.move_entity("P1", move=move)
                        # If a move has been made, then the agents should be able to percept the
                        # surroundings.
                        percept = True
                    except GameState.IlegalMove as exception:
                        print(exception)
            # Executes only if game is paused
            if e.type == p.MOUSEBUTTONDOWN:
                if game_paused and input_rect.collidepoint(e.pos):
                    text_active = True
                    response_active = False
                else:
                    text_active = False
                    response_active = True

        # Check if game has ended directly after move has been made...
        is_end = gs.is_terminate()
        if is_end != 0:
            print("\nEnd:")
            if is_end == 1:
                print("You have made a great job surviving and hunting the werewolves. The "
                      "villagers consider you a hero and owe you eternal gratitude.")
            elif is_end == 2:
                print("You returned to the town with empty hands. You are a coward, your reputation has fallen. The "
                      "villagers despise you.")
            else:
                print("You were killed. Be more cautious in your next life.")
            running = False

        # If agents can percept and game is still on...
        if percept and running:
            # Previous player position
            x_prev, y_prev = new_state[0], new_state[1]
            percepts = gs.percepts((x_prev, y_prev))
            # Get coordinates of recommendation
            recommendation = recommender.recommend(gs, percepts)

            smell_active = True if percepts["smell"] else False
            hear_active = True if percepts["hear"] else False
            see_active = True if percepts["see"] else False

            d_clauses, w_clauses, v_clauses = get_expr_from_percepts(percepts, gs.where_is_player(), gs.board)
            # Check if there was a battle
            if new_state[2]:
                entities_found.append((new_state[0], new_state[1]))
                entity: str = gs.board[new_state[0], new_state[1]]
                if entity.startswith("W"):
                    recommender.modify_prior("werewolf", {(new_state[0], new_state[1]): 1})
            else:
                d_agent.tell(d_clauses, f"~D[{x_prev},{y_prev}]")
                w_agent.tell(w_clauses, f"~W[{x_prev},{y_prev}]")
                v_agent.tell(v_clauses, f"~V[{x_prev},{y_prev}")
            # Reset value to False
            percept = False

        clock.tick(MAX_FPS)
        p.display.flip()
