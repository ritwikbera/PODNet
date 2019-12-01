#!/usr/bin/env python3

import sys
import numpy as np
import gym
import time
import datetime as dt
from matplotlib import pyplot as plt
import gym_minigrid
import pandas as pd

from gym import spaces

class ReducedObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding that returns as
    observation only agent (x,y) position and orientation.
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        return np.array([env.agent_pos[0],env.agent_pos[1],env.agent_dir])

def main():
    # Start CSV log file
    # Structure: episode, time step, action, obs
    time_now = dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_file = open('data/{}_log.csv'.format(time_now), 'w')
    log_file.write('episode,time_step,action,x,y,heading\n')

    # Load the gym environment
    # Add wrapper to modify observation
    # env = ReducedObsWrapper(gym.make("MiniGrid-Empty-Random-6x6-v0"))
    env = ReducedObsWrapper(gym.make("MiniGrid-DoorKey-16x16-v0"))
    env.episode_count = 0

    def resetEnv():
        # Reset environment
        obs = env.reset()
        if hasattr(env, 'mission'):
            print('[*] EPISODE {} | Mission: {}'.format(
                env.episode_count,env.mission))

        # Log initial observation
        action = 0
        log_file.write('{},{},{},{}\n'.format(
            env.episode_count, env.step_count, int(action),
            str(obs.flatten().tolist())[1:-1]))

        return obs

    obs = resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            obs = resetEnv()
            print('initial obs: ', obs)
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done

        # Screenshot funcitonality
        elif keyName == 'ALT':
            screen_path = '{}_{}.png'.format(time_now, env.step_count)
            print('saving screenshot "{}"'.format(screen_path))
            pixmap = env.render('pixmap')
            pixmap.save(screen_path)
            return

        else:
            print("unknown key %s" % keyName)
            return

        # Step action
        obs, reward, done, info = env.step(action)
        print('Obs {} | Action {} '.format(obs, action))
        
        # Write to log file
        log_file.write('{},{},{},{}\n'.format(
            env.episode_count, env.step_count, int(action),
            str(obs.flatten().tolist())[1:-1]))


        if done:
            print('Done after {} steps!'.format(env.step_count))
            env.episode_count += 1
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            # Close log file
            log_file.close()

            # Closes render
            break

if __name__ == "__main__":
    main()
