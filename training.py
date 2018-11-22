try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

import gym
from gym.spaces import Box, Discrete
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
from vrep_env import VrepEnv

import errno
import os
from datetime import datetime

from actor_net import ActorNet
from critic_net import CriticNet

import argparse

from api import vrep


episodes=100000

def main():
    env=VrepEnv()


    agent = DDPG(env)
    
    
    exploration_noise = OUNoise(env.action_space.shape[0])
    counter=0
    reward_per_episode = 0
    total_reward=0
    num_states=env.observation_space.shape[0]
    num_actions=env.action_space.shape[0]
    print("Number of states:", num_states)
    print("Numver of Actions:", num_actions)


    reward_st = np.array([0])

    for i in range(episodes):
            print("==== starting episode no:",i,"====")
            observation = env.reset()

            print(observation)

            reward_per_episode = 0

            step = 0
            while True:
                x = observation
                action = agent.evaluate_actor(np.reshape(x,[1,num_states]))

                noise = exploration_noise.noise()

                action = action[0] + noise


                observation,reward=env.step(action)

                agent.add_experience(x,observation,action,reward,step)

                if counter > 64:
                    agent.train()
                reward_per_episode += reward
                counter+=1

                if env.finishCheck(observation):
                    print('EPISODE: ',i,'Steps: ',step,' Total Reward: ', reward_per_episode)

                    exploration_noise.reset()
                    reward_st = np.append(reward_st,reward_per_episode)
                    np.savetxt('episode_reward.txt',reward_st, newline="\n")

                    if i % 100 == 0:
                        print('save')
                        agent.save.model()
                    
                    break
                
                step += 1
        
            total_reward += reward_per_episode
            print("Average reward per episode {}".format(total_reward / episodes)  )

if __name__ == '__main__':
    main()