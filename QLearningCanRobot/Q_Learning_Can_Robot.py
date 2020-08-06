#Q_Learning Robot
#Authored by David Hogan
#davhogan@pdx.edu
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

Q_ROWS = 512
NUM_ACTIONS = 5
PICK_UP = 0
NORTH = 1
SOUTH = 2
EAST = 3
WEST = 4
Position = namedtuple("Position", "x y")

#Used to lookahead at next tile
#Returns as a string of a given tile as a wall, empty or contains a coin
def get_tile(grid, x, y):
    if x < 0 or x > 9 or y < 0 or y > 9:
        return "Wall"
    else:
        if grid[x][y] == 0:
            return "Empty"
    return "Coin"

#Used to lookahead at next tile
#Returns as a binary representation of
#a given tile as a wall, empty or contains a coin
#0,0 = empty, 0,1 = coin 1,0 = Wall
def get_tile_binary(grid, x, y):
    if x < 0 or x > 9 or y < 0 or y > 9:
        return [1, 0]
    else:
        if grid[x][y] == 0:
            return [0, 0]
    return [0, 1]

#Returns the next tile in a given direction
#As a string reprsentation
def look_ahead(grid, curr_pos, direction):
    if direction == 'N' or direction == NORTH:
        return get_tile(grid, curr_pos.x - 1, curr_pos.y)
    elif direction == 'S' or direction == SOUTH:
        return get_tile(grid, curr_pos.x + 1, curr_pos.y)
    elif direction == 'E' or direction == EAST:
        return get_tile(grid, curr_pos.x, curr_pos.y + 1)
    elif direction == 'W' or direction == WEST:
        return get_tile(grid, curr_pos.x, curr_pos.y - 1)
    return "Error"

#Returns the next tile in a given direction
#As a binary reprsentation
def look_ahead_bin(grid, curr_pos, direction):
    if direction == 'N' or direction == NORTH:
        return get_tile_binary(grid, curr_pos.x - 1, curr_pos.y)
    elif direction == 'S' or direction == SOUTH:
        return get_tile_binary(grid, curr_pos.x + 1, curr_pos.y)
    elif direction == 'E' or direction == EAST:
        return get_tile_binary(grid, curr_pos.x, curr_pos.y + 1)
    elif direction == 'W' or direction == WEST:
        return get_tile_binary(grid, curr_pos.x, curr_pos.y - 1)
    return [1, 1]

#Returns the state of the robot's current position
#The state is the current tile contains
#As well as what the tiles to the robot's N,S,E and W contain
def get_state(grid, curr_pos):
    state = []
    current = grid[curr_pos.x][curr_pos.y]
    north = look_ahead_bin(grid, curr_pos, 'N')
    south = look_ahead_bin(grid, curr_pos, 'S')
    east = look_ahead_bin(grid, curr_pos, 'E')
    west = look_ahead_bin(grid, curr_pos, 'W')
    state.append(current)

    state.extend(north)
    state.extend(south)
    state.extend(east)
    state.extend(west)

    return state

#Converts a given binary state to an integer
#The returned value is the row the state corresponds to on the q-table
def get_state_row(state):
    state_row = 0
    for i in range(0, len(state)):
        if state[i] == 1:
            state_row += 2 ** i
    return state_row

#Moves the robot based on the given direction
#Returns the robots new x and y coordinates
def move(grid, curr_pos, direction):
    # If move would run robot into wall do nothing
    if look_ahead(grid, curr_pos, direction) == "Wall":
        return curr_pos.x, curr_pos.y
    # Move the robot in the desired direction
    if direction == 'N' or direction == NORTH:
        return curr_pos.x - 1, curr_pos.y
    elif direction == 'S' or direction == SOUTH:
        return curr_pos.x + 1, curr_pos.y
    elif direction == 'E' or direction == EAST:
        return curr_pos.x, curr_pos.y + 1
    elif direction == 'W' or direction == WEST:
        return curr_pos.x, curr_pos.y - 1
    return curr_pos.x, curr_pos.y


# Picks up the can(if there is one) at the current position
def pick_up(grid, curr_pos):
    grid[curr_pos.x][curr_pos.y] = 0
    return grid

#Takes the action given
#Either the robot tries to picl up a can
#Or it tries to move in the given direction
#The updated grid and robot's position are returned
def take_action(grid, position, action):
    pos_x = position.x
    pos_y = position.y
    #Pick Up Can
    if action == PICK_UP:
        grid = pick_up(grid, position)
    #Move Robot
    else:
        pos_x, pos_y = move(grid, position, action)

    return grid, pos_x, pos_y

#Checks the action the robot is trying to make
#Returns the reward for a given action
def check_action(grid, position, action):
    # action is pickup
    if action == 0:
        #Tile contains a coin
        if get_tile(grid, position.x, position.y) == "Coin":
            return 10
        else:
        #Tile is empty
            return -1
    # action is move
    #Ran into a wall
    if look_ahead(grid, position, action) == "Wall":
        return -5
    #Moved into a new space
    return 0

#Trains the robot for n epoch with m steps for each epoch
#Robot explores and exploits the given grid and updates
#the q_table based on the results of the actions taken for a given state.
#Returns the q-table after being trained
def q_training(n, m):

    q_table = np.zeros((Q_ROWS, NUM_ACTIONS))
    learn_rate = 0.2
    epsilon = 0.1
    gamma = .9
    total_rewards = []
    #Epochs
    for i in range(0, n):
        tot_reward = 0
        #Create a 10x10 grid filled randomly filled with cans
        grid = np.random.randint(2, size=(10, 10))

        #Place robot at random tile in grid
        rand_x = np.random.randint(0, 10)
        rand_y = np.random.randint(0, 10)
        current_pos = Position(rand_x, rand_y)

        #Get the state of the current state of the robot
        state = get_state(grid, current_pos)
        state_row = get_state_row(state)
        #Single training for generated grid
        for j in range(0, m):
            #Try for a random action
            if epsilon > np.random.uniform(0, 1):
                action = np.random.randint(0, 5)
            #Take the best action found so far
            else:
                action = np.argmax(q_table[state_row])
            #Get current q_value
            curr_q_val = q_table[state_row][action]
            curr_state_row = state_row

            reward = check_action(grid, current_pos, action)
            tot_reward = tot_reward + reward
            #Have the robot take chosen action
            grid, current_pos_x, current_pos_y = take_action(grid, current_pos, action)
            current_pos = Position(current_pos_x, current_pos_y)
            #get values for updating q_table
            next_state = get_state(grid, current_pos)
            state_row = get_state_row(next_state)
            max_q_val = max(q_table[state_row])

            #Update the q_table
            q_table[curr_state_row][action] = curr_q_val + learn_rate*(reward + gamma * max_q_val - curr_q_val)

            #Make a jump to a random tile in the grid
            if j % 50 == 0:
                rand_x = np.random.randint(0, 10)
                rand_y = np.random.randint(0, 10)
                current_pos = Position(rand_x, rand_y)
        #Total_reward for an epoch
        total_rewards.append(tot_reward)

        #Reduce exploration factor
        if i % 50 == 0:
            epsilon -= .002
            print(i)

        if epsilon < 0:
            epsilon = 0

    #Plot training data
    total_rewards_plt = []
    for i in range(0, len(total_rewards)):
        if i % 100 == 0:
            total_rewards_plt.append(total_rewards[i])
    plt.plot(total_rewards_plt)
    plt.suptitle("Training\n Epilson Decrease Rate: -.002")
    plt.show()

    return q_table

#Uses the q-table generated from testing to traverse a new environment
#Tries n new environments with m steps
#Returns the average reward and standard deviation for all of the attempts.
def q_testing(q_table, n, m):

    epsilon = 0.1
    total_rewards = []
    #epochs
    for i in range(0, n):
        tot_reward = 0
        grid = np.random.randint(2, size=(10, 10))

        rand_x = np.random.randint(0, 10)
        rand_y = np.random.randint(0, 10)
        current_pos = Position(rand_x, rand_y)

        state = get_state(grid, current_pos)
        state_row = get_state_row(state)
        #Steps through environment
        for j in range(0, m):
            if epsilon > np.random.uniform(0, 1):
                action = np.random.randint(0, 5)
            else:
                action = np.argmax(q_table[state_row])
            #Check and take given action to traverse the environment
            reward = check_action(grid, current_pos, action)
            tot_reward = tot_reward + reward

            grid, current_pos_x, current_pos_y = take_action(grid, current_pos, action)
            current_pos = Position(current_pos_x, current_pos_y)

            next_state = get_state(grid, current_pos)
            state_row = get_state_row(next_state)
            #Random jump to new tile in q-table
            if j % 50 == 0:
                rand_x = np.random.randint(0, 10)
                rand_y = np.random.randint(0, 10)
                current_pos = Position(rand_x, rand_y)
        #total_reward for an epoch
        total_rewards.append(tot_reward)

    avg = np.average(total_rewards)
    std = np.std(total_rewards)
    return avg, std

#Train a robot
q_table = q_training(5000, 200)
#Robot moves in new environments based on the results from its training
avg, std = q_testing(q_table, 5000, 200)
print("Test Average =", avg)
print("Test Standard Deviation =", std)
