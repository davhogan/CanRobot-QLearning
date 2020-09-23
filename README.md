# CoinRobot-QLearning
A robot learns to pick up coins that are randomly placed on a grid. The robot uses Q-Learning to figure out the best approach for traversing the grid and collecting coins.

A 10 x 10 grid is randomly filled with coins. Each tile of the grid will either be empty or contain a coin. The robot is then placed on a random tile in the grid. 
The robot knows what is conatined in the space it is currently occupying as well as what is to the north, south, east and west of its current location. 
The robot's aviable actions are: pick up, do nothing as well as to make a move north, south, east or west.
The robot is given a reward based on its action:
- Pick up a coin : 10 points
- Do nothing : 0 points
- Perform the pick up action in an empty tile: -1 point
- Run into a wall: -5 points

At first the robot traverses the grid by choosing a random action. As these random actions are performed the result is kept in a qtable where each enrty in the table represents a different state the robot was in when the action was chosen. 

The robot takes 200 actions in each attempt at traversing the grid.

After a period of trials using random choices the robot switches to using the qtable to make the most ideal move based on its previous experiences. The results of these choices are still recorded in the qtable. 

The final result of the actions attempted are recodered and displayed after each traversal through the grid.  
