"""
This file contains the main engine of the game. All game mechanics are implemented on this file.
"""
from random import randint
import numpy as np

from itertools import count
from heapq import heappop, heappush
from typing import Dict

from constants import PINF, BOARD_WIDTH, BOARD_HEIGHT, NUMBER_OF_PORTALS, NUMBER_OF_WEREWOLVES,\
    BOARD_GENERATION_ATTEMPTS, ENTITY_PERCEPT, THRESHOLD_MEAN, THRESHOLD
from agents import *
from entities import *

__all__ = ["GameState", "get_expr_from_percepts", "Recommender"]


def out_of_bounds(matrix, coordinates):
    # Negative coordinates
    if coordinates[0] < 0 or coordinates[1] < 0:
        return True

    # Coordinates exceeding matrix dimensions
    x, y = matrix.shape
    if coordinates[0] > (x - 1) or coordinates[1] > (y - 1):
        return True
    return False


def generate_board(attempts: int = BOARD_GENERATION_ATTEMPTS):
    # Create empty board and empty set of mobs
    board = np.full(shape=(BOARD_HEIGHT, BOARD_WIDTH), fill_value="--")
    objects = dict()

    # Spawning players
    board[0, 0] = "P1"
    objects["P1"] = Player(id_=1)

    # Generate village
    while True:
        if board[x := randint(0, BOARD_HEIGHT - 1), y := randint(0, BOARD_WIDTH - 1)] == "--":
            board[x, y] = f"VI"
            exit_position = [(x, y)]
            break

    # Spawning werewolf
    assert 0 < NUMBER_OF_WEREWOLVES < (BOARD_HEIGHT * BOARD_WIDTH - NUMBER_OF_PORTALS - 2)

    werewolf_positions = []
    for werewolf in range(NUMBER_OF_WEREWOLVES):
        werewolf_id = werewolf + 1
        while True:
            if board[x := randint(0, BOARD_HEIGHT - 1), y := randint(0, BOARD_WIDTH - 1)] == "--":
                board[x, y] = f"W{werewolf_id}"
                objects[f"W{werewolf_id}"] = Werewolf(werewolf_id)
                werewolf_positions.append((x, y))
                break

    # Spawning "upside down portals"
    board = generate_portals(board, objects, werewolf_positions, exit_position, attempts)
    return board, objects


def generate_portals(matrix, objects, wer_pos, ex_pos, attempts):
    """
    Generates a valid set of portals in the board, avoiding unwinnable game states. This function
    may not find an appropriate distribution of the portals within the number of attempts specified
    """
    # Set a number of tries. Not deterministic, can be replaced with TSP implementation.
    for _ in range(attempts):
        m = matrix.copy()

        for portal in range(NUMBER_OF_PORTALS):
            portal_id = portal + 1
            while True:
                if m[x := randint(0, BOARD_HEIGHT - 1), y := randint(0, BOARD_WIDTH - 1)] == "--":
                    m[x, y] = f"D{portal_id}"
                    objects[f"D{portal_id}"] = DemonPortal(portal_id)
                    break

        # Search the shortest path. If shortest path exists, then at least one path that
        # joins all objectives exist.
        def is_valid_dem_path(arr, x_coord, y_coord):
            """
            A valid path joining each werewolf to the player is a path without having to necessarily
            access the village nor portals.
            """
            return not arr[x_coord, y_coord].startswith("D") and not m[x_coord, y_coord] == "VI"

        def is_valid_ex_path(arr, x_coord, y_coord):
            """
            A valid path joining a werewolf to the village is a path without having to necessarily
            access any portal.
            """
            return not arr[x_coord, y_coord].startswith("D")

        # If a path exists joining each werewolf to the player without having to access the village
        # and a path from any werewolf to the village exists, then the current game board is valid.
        # Portals serve as nodes blocking a series of paths.
        if find_shortest_path_length(m, (0, 0), wer_pos, is_valid_dem_path) == PINF or (
                find_shortest_path_length(m, wer_pos[0], ex_pos, is_valid_ex_path) == PINF):
            continue
        return m


def find_shortest_path_length(matrix, source, destinations, valid=lambda *args: True):
    """
    Finds the shortest paths from a source to each destination. However, the resulting paths will not
    necessarily have to contain each destination (this is not a TSP implementation). This function uses
    an unweighted version of Dijkstra's algorithm.
    :parameter matrix: Matrix containing nodes and paths.
    :parameter source: Starting point of the Algorithm.
    :parameter destinations: Array containing all destinations.
    :parameter valid: Function that determines if a node is valid or not.
    :return: Length of the obtained path.
    """
    # Dictionary of distances to each node
    distances = {}
    # Set all distances to infinity
    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            distances[(i, j)] = PINF

    # Set distance to source node to 0
    distances[source] = 0

    # Initialize heap queue and path and visited dictionaries. Path dictionary contains
    # the path to each node
    queue, visited, path = [], {}, {source: [source]}

    heappush(queue, (0, next(counter := count()), source))

    # Iterate over queue until all destinations have been visited
    while queue and (destinations_left := destinations.copy()):
        # Pop first element of the queue
        curr_distance, _, curr_node = heappop(queue)
        if curr_node in visited:
            continue
        visited[curr_node] = True

        # Remove destination if visited
        if curr_node in destinations:
            destinations_left.remove(curr_node)

        # Iterate through neighbours
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbour = (curr_node[0] + dx, curr_node[1] + dy)
            if not out_of_bounds(matrix, (neighbour[0], neighbour[1])):
                # Skip neighbour if not valid
                if not valid(matrix, neighbour[0], neighbour[1]):
                    continue
                distance = curr_distance + 1
                # If new distance is lower than previous new distance, modify
                # neighbour's distance
                if distances[neighbour] > distance:
                    distances[neighbour] = distance
                    # Push neighbour to queue
                    heappush(queue, (distance, next(counter), neighbour))
                    path[neighbour] = path[curr_node] + [neighbour]
    return get_distance(destinations, path)


def get_distance(destinations, path):
    """
    Given a dictionary of path to each node, it sums the length of each path reaching
    the destinations. If a destination is not on the dictionary, then the distance to
    this destination is infinite.
    """
    visited = {}
    total_distance = 0

    # Iterate over the coordinates to search
    for coord in destinations:
        if coord in visited:
            continue
        visited[coord] = True
        if coord not in path:
            return PINF
        total_distance += len(path[coord]) - 1
    return total_distance


class GameState:
    def __init__(self):
        """
        This class represent the current state of the game.
        """
        self.board, self.entities = generate_board()

    class IlegalMove(Exception):
        """
        Exception raised when a specific move is not allowed within the game rules.
        Example: A movement to a coordinate out of bounds.
        """
        def __init__(self, message="Specified move is ilegal"):
            super().__init__(message)

    def move_entity(self, entity_str: str, move: str):
        e = self.entities[entity_str]
        coords = np.where(self.board == entity_str)
        x_curr, y_curr = int(coords[0]), int(coords[1])
        moveset = e.MOVESET

        # Check if move is legal. Currently, the only illegal moves are being out of bounds or
        # not being on the entity's moveset.
        if move in moveset and not out_of_bounds(self.board, (x := x_curr + moveset[move][0],
                                                              y := y_curr + moveset[move][1])):
            if (d_str := self.board[x, y]) in self.entities:
                d = self.entities[d_str]
                # If entity attacks...
                if moveset[move][2] and not isinstance(d, DemonPortal):
                    x, y = self.battle(attacker=e, defender=d, position=(x_curr, y_curr, x, y))
                # Otherwise...
                else:
                    x, y = self.battle(attacker=d, defender=e, position=(x, y, x_curr, y_curr))
                return x, y, True
            else:
                # If entity attacks...
                if moveset[move][2]:
                    e.hit(target=None)
                # Finally, move entity
                self.board[x, y] = entity_str
                self.board[x_curr, y_curr] = "--"
                return x, y, False
        else:
            raise self.IlegalMove

    def where_is_player(self):
        """
        Returns the player position.
        """
        x, y = np.where(self.board == "P1")
        return int(x), int(y)

    def percepts(self, position):
        """
        Returns a dictionary of percepts given a position. If a werewolf is adjacent to the
        given position, a player in this position will hear some "grunting" noises. If a
        demon portal If a werewolf is adjacent, a player in this position will smell like
        something is rotting. If the village is nearby, the player will be able to see some
        distant lights.
        """
        senses = {sense: False for sense in ENTITY_PERCEPT.values()}

        x, y = position
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            adj_x, adj_y = x + dx, y + dy
            if not out_of_bounds(self.board, (adj_x, adj_y)):
                value = self.board[adj_x, adj_y]
                if value.startswith("W"):
                    senses["hear"] = True
                elif value.startswith("D"):
                    senses["smell"] = True
                elif value.startswith("V"):
                    senses["see"] = True
        return senses

    def battle(self, attacker, defender, position):
        """
        Initiates a battle between an attacker and a defender. The position is a tuple of
        4 integers where the first two stand for the coordinates of the attacker and the
        other two of the defender. If the attacker kills the defender then, the attacker
        takes the defenders place, otherwise no movement will happen.
        """
        ax_curr, ay_curr, ax, ay = position
        attacker.hit(target=defender)
        if not defender.alive():
            print(f"{defender} has been killed by {attacker}.")
            self.board[ax, ay] = str(attacker)
            self.board[ax_curr, ay_curr] = "--"
            return ax, ay
        return ax_curr, ay_curr

    def are_werewolves(self):
        """
        Checks if there are any werewolves left in the bord.
        :return: True if there are, False otherwise.
        """
        return bool(np.sum(np.char.startswith(self.board, "W")))

    def is_terminate(self):
        """
        Checks if the game has ended or not, according to a set of "goal states".
        :return: 0 if the game has not ended, 1 if player wins and 2 or 3 otherwise.
        """
        # If there is no player was killed or had his weapon broken and there are still
        # werewolves around, player loses
        if self.board[self.board == "P1"].shape[0] == 0 or (
                self.are_werewolves() and self.entities["P1"].weapon.broken()):
            return 3
        # If there is no village left...
        elif self.board[self.board == "VI"].shape[0] == 0:
            # If werewolves were killed, player wins
            if not self.are_werewolves():
                return 1
            # Otherwise player loses
            else:
                return 2
        # Otherwise the game has not ended yet
        return 0


def get_expr_from_percepts(percepts: dict, position: tuple, board):
    x, y = position

    expr_smell = ""
    expr_hear = ""
    expr_see = ""

    smell_negated = "" if percepts["smell"] else "~"
    hear_negated = "" if percepts["hear"] else "~"
    see_negated = "" if percepts["see"] else "~"

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        adj_x, adj_y = x + dx, y + dy
        if not out_of_bounds(board, (adj_x, adj_y)):
            expr_smell += smell_negated + f"D[{adj_x},{adj_y}] |"
            expr_hear += hear_negated + f"W[{adj_x},{adj_y}] |"
            expr_see += see_negated + f"V[{adj_x},{adj_y}] |"

    return expr_smell[0:-2], expr_hear[0:-2], expr_see[0:-2]


class Recommender:
    def __init__(self, gs: GameState, version: str = "bayesian"):
        n, m = gs.board.shape

        if version == "bayesian":
            p = 0.90
            q = 0.05
        elif version == "logical":
            p = 0.99999
            q = 0.00001
        else:
            raise ValueError(f'Sorry, this version "{version}" is not yet implemented.')

        self.agents = {"werewolf": BayesianAgent(n=n, m=m, success=NUMBER_OF_WEREWOLVES, p=p, q=q),
                       "demon_portal": BayesianAgent(n=n, m=m, success=NUMBER_OF_PORTALS, p=p, q=q),
                       "village": BayesianAgent(n=n, m=m, success=1, p=p, q=q)}

    def modify_prior(self, agent: str, dictionary):
        return self.agents[agent].modify_prior(dictionary)

    def recommend(self, gs: GameState, percepts: Dict[str, bool]):
        x, y = gs.where_is_player()
        probabilities = {}

        for agent_type, agent_obj in self.agents.items():
            agent_obj.modify_prior({(x, y): 0})
            feel = percepts[ENTITY_PERCEPT[agent_type]]
            probabilities[agent_type] = dict()
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                adj_x, adj_y = x + dx, y + dy
                if not out_of_bounds(gs.board, (adj_x, adj_y)):
                    # Step 1: Modify likelihood based on percepts
                    agent_obj.modify_likelihood0(adj_x, adj_y, feel)
                    agent_obj.modify_likelihood1(adj_x, adj_y, feel)
                    # Step 2: Calculate posteriors
                    posterior = agent_obj.calculate_posterior(adj_x, adj_y)
                    probabilities[agent_type][(adj_x, adj_y)] = posterior
            # Step 3: Update prior
            agent_obj.modify_prior(probabilities[agent_type])
        # print(probabilities)    # Remove the hashtag to display probabilities

        if gs.are_werewolves():
            # Search minimum probability
            mean_values = {coord: (sum(probabilities[key].get(coord, 0) for key in probabilities
                                       ) / len(probabilities)) for coord in
                           set.union(*(set(val) for val in probabilities.values()))}
            min_coord = min(mean_values, key=mean_values.get)

            for value in mean_values.keys():
                if self.agents['demon_portal'].prior[value[0], value[1]] == 0:
                    continue
                if mean_values[value] < THRESHOLD_MEAN:
                    min_coord = value
            return min_coord
        else:
            def custom_key(coord):
                if probabilities['demon_portal'][coord] < THRESHOLD:
                    return probabilities['village'][coord], -probabilities['demon_portal'][coord]
                else:
                    return -PINF, -PINF

            best_coordinate = max(probabilities['village'], key=custom_key)
            return best_coordinate
