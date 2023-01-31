# Dungeons_and_Dragons: The Hunt
## Description
This is an unofficial videogame based on the famous tabletop role-playing game "Dungeons and Dragons", created for academic purposes. The main reason behind this project was to implement simple AI (Knowledge-based agent and Bayesian probabilistic agent) ingame.

## Objective
This game was inspired by the Wumpus World game, famous example to illustrate knowledge representation in AI. By default, there are 3 demon portals, 1 village and 1 werewolf. The latter is the player's main objective to kill before returning to the village. If the player is killed or returns to the village before killing the werewolf he loses. The trick behind this game is that the night has fallen and it has gotten very dark, so the player must really on his senses to complete his task.

## How to run it:
First and foremost install the packages from requirements.txt. Run the file "main.py" inside the "src" directory.

## Structure:
Inside the "src" directory there are several files, "main.py" being the main file of them all. It contains the basic app settings, game loop and graphical engine. Inside "engine.py" you may find the basic algorithms and classes of the game. Logical and Probabilistic agents, and game entity objects are inside "agents.py" and "entities.py" respectively. The user may configure "constants.py" (to personalize their gameplay) as they will following instructions commented in the file.

## Images:
![alt text](https://github.com/SeroviICAI/Dungeons_and_Dragons_Game/blob/master/images/game_screenshot.jpg)

## More information:
You can find more information about this project in "memoria.pdf" (this file is in spanish) and code documentation (english).
