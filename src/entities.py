"""
This file contains all entity classes that will be used by the game engine. Entities are objects with
their own properties and game mechanics. Each entity has a type and id that identifies it.
"""
from random import randint
from typing import Dict, Tuple, Union
from constants import WEAPONS, PLAYER_HEALTH, WEREWOLF_HEALTH, PORTAL_HEALTH, PLAYER_STARTER_WEAPON

__all__ = ["Player", "Werewolf", "DemonPortal"]


class Weapon:
    def __init__(self, name: str, owner: object):
        if name not in WEAPONS:
            raise ValueError(f'Not valid weapon "{name}".')
        self.owner = owner

        weapon_dict = WEAPONS[name]
        self.durability = weapon_dict["durability"]
        self.damage = weapon_dict["damage"]
        self.miss_chance = weapon_dict["miss_chance"]

    def broken(self):
        return self.durability < 1

    def hit(self, target):
        if not self.broken():
            if randint(0, 100) >= (self.miss_chance * 100):
                self.durability -= 1
                target.health -= self.damage
                print(f"{self.owner} dealt {self.damage} damage to {target}.")
            else:
                self.miss(target=target)
            if self.durability == 0:
                print(f"{self.owner}'s weapon broke.")
        else:
            print(f"{self.owner}'s weapon is broken. "
                  f"{self.owner} dealt 0 damage to {target}.")

    def miss(self, target=None):
        self.durability -= 1
        if target is None:
            print(f"{self.owner} hit the ground.")
        else:
            print(f"{self.owner} missed {target}.")

        if self.durability == 0:
            print(f"{self.owner}'s weapon broke.")


class Entity(object):
    """
    Movable in game objects with properties and actions they can perform.
    """
    _MOVESET = Union[Dict[str, Tuple[int, int, bool]], None]

    def __str__(self):
        return f"{self.type}{self.id}"

    def __init__(self, id_: int, type_: str, health: int, weapon: Weapon = None, moveset: _MOVESET = None):
        if moveset is None:
            moveset = self.empty_moveset()
        self.id = id_
        self.type = type_
        self.health = health
        self.weapon = weapon

    @staticmethod
    def empty_moveset():
        return {}

    def alive(self):
        return self.health > 0

    def hit(self, target: "Entity" or None = None):
        if self.weapon is not None:
            if target is None:
                return self.weapon.miss()
            return self.weapon.hit(target)
        else:
            return print(f'{str(self)} has no weapons.')


class Player(Entity):
    """
    Entity representing the player in game.
    """
    TYPE = "P"
    MOVESET = {"UP": (-1, 0, False), "UP+ATTACK": (-1, 0, True),
               "RIGHT": (0, 1, False), "RIGHT+ATTACK": (0, 1, True),
               "LEFT": (0, -1, False), "LEFT+ATTACK": (0, -1, True),
               "DOWN": (1, 0, False), "DOWN+ATTACK": (1, 0, True)}
    WEAPON = PLAYER_STARTER_WEAPON

    def __init__(self, id_: int = 1):
        super().__init__(id_=id_, type_=self.TYPE,
                         health=PLAYER_HEALTH,
                         weapon=Weapon(self.WEAPON, self),
                         moveset=self.MOVESET)


class Werewolf(Entity):
    """
    Entity representing the werewolves in game.
    """
    WEAPON = "PAW"
    MOVESET = None
    TYPE = "W"

    def __init__(self, id_: int = 1):
        """
        Werewolves are entities that may attack the player and reduce players health by 1 HP.
        :parameter id_: Unique identifier of the werewolf.
        """
        super().__init__(id_=id_,
                         type_=self.TYPE,
                         health=WEREWOLF_HEALTH,
                         weapon=Weapon(self.WEAPON, self),
                         moveset=self.MOVESET)


class DemonPortal(Entity):
    """
    Entity representing the demon portals in game.
    """
    WEAPON = "PORTAL"
    MOVESET = None
    TYPE = "D"

    def __init__(self, id_: int = 1):
        """
        Demon portals are entities without a move set, which are inmortal and perform infinite
        damage to a target.
        :parameter id_: Unique identifier of the portal
        """
        super().__init__(id_=id_,
                         type_=self.TYPE,
                         health=PORTAL_HEALTH,
                         weapon=Weapon(self.WEAPON, self),
                         moveset=self.MOVESET)