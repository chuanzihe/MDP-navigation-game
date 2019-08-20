# MDP-navigation-game
Python implementation of Value Iteration in a navigation MDP game.

## Problem Description

You are the CTO of a new startup company, SpeedRacer, and you want your autonomous cars to navigate throughout the city of Los Angeles. The cars can move North, South, East, or West (see directions to the right). The city can be represented in a grid, as below:

| 0,0 | 1,0 | 2,0 | 3,0 | 4,0 |
|-----|-----|-----|-----|-----|
| 0,1 |     |     |     |     |
| 0,2 |     |     |     |     |
| 0,3 |     |     |     |     |
| 0,4 |     |     |     | 4,4 |

There will be some obstacles, such as buildings, road closings, etc. If a car crashes into a building or road closure, SpeedRacer has to pay $100. You know the locations of these, and they will not change over time. You also spend $1 for gas each time you move. The cars will start from a given SpeedRacer parking lot, and will end at another parking lot. When you arrive at your destination parking lot, you will receive $100. **Your goal is to make the most money over time with the greatest likelihood.** Your cars have a faulty turning mechanism, so they have a chance of going in a direction other than the one suggested by your model. They will go in the correct direction 70% of the time (10% in each other direction, including along borders).

The first part of your task is to design an algorithm that determines where your cars should try to go in each city grid location given your goal of making the most money. Then, to make sure that this is a good algorithm when you present it to the rest of your board, you should simulate the car moving through the city grid. To do this, you will use your policy from your start location. You will then check to see if the car went in the correct direction using a random number generator with specific seeds to make sure you can reproduce your output. You will simulate your car moving through the city grid 10 times using the random seeds 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9. You will report the mean over these 10 simulations as an integer after using the floor operation (e.g., numpy.floor(meanResult)). An example of this process is given in detail below.

Input: The file input.txt in the current directory of your program will be formatted as follows:

- First line: strictly positive 32-bit integer s, size of grid [grid is a square of size sxs] 
- Second line: strictly positive 32-bit integer n, number of cars 
- Third line: strictly positive 32-bit integer o, number of obstacles 
- Next o lines: 32-bit integer x, 32-bit integer y, denoting the location of obstacles 
- Next n lines: 32-bit integer x, 32-bit integer y, denoting the start location of each car 
- Next n lines: 32-bit integer x, 32-bit integer y, denoting the terminal location of each car 

Output:

- n lines: 32-bit integer, denoting the mean money earned in simulation for each car, integer result of floor operation
