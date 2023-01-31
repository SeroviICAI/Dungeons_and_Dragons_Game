"""
This file contains all constants that will be used by the game engine. They can manually be modified, but not
in game.
"""
from numpy import inf

PINF = inf

# GAME RULES #
PLAYER_HEALTH = 3
PLAYER_STARTER_WEAPON = "AXE"

WEREWOLF_HEALTH = 1
PORTAL_HEALTH = PINF  # Do not modify this

# Do not change keys withing weapons (durability, damage,...)
WEAPONS = {
    "AXE": {
        "durability": 3,
        "damage": 1,
        "miss_chance": 0.2
    },
    "PAW": {
        "durability": PINF,
        "damage": 1,
        "miss_chance": 0.
    },
    "PORTAL": {
        "durability": PINF,
        "damage": PINF,
        "miss_chance": 0.
    }
}

# Do not modify this
ENTITY_PERCEPT = {
    "werewolf": "hear",
    "demon_portal": "smell",
    "village": "see"
}

# BOARD GENERATION #
BOARD_HEIGHT = BOARD_WIDTH = 5  # They must be the same size
BOARD_GENERATION_ATTEMPTS = 100000  # Low values not recommended

NUMBER_OF_WEREWOLVES = 1  # More than zero
NUMBER_OF_PORTALS = 3

# GRAPHICS #
map_dimension = 800
SQ_SIZE = map_dimension // BOARD_HEIGHT
SCREEN_HEIGHT = BOARD_HEIGHT * SQ_SIZE
SCREEN_WIDTH = BOARD_WIDTH * SQ_SIZE
MAX_FPS = 15

# LOGICAL AGENT #
MAX_ATTEMPTS_EXPR_PROCCESSING = 100  # Very low values (<5) not recommended

# RECOMMENDER #
RECOMMENDER_TYPE = "bayesian"
# This is the maximum a recommended probability will be, meaning no coordinate with
# a higher mean of probabilities than THRESHOLD_MEAN. High values make the recommender
# less reliable, low value make it too safe for the player (it may end up recommending
# visited coordinates and creating loops). Recommended: between 0.13 and 0.20 for this
# configuration.
THRESHOLD_MEAN = 0.13

# This is the maximum a recommended coordinate's demon_portal probability will be after
# all werewolves have been exterminated.
THRESHOLD = 0.75
