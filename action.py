from __future__ import print_function, absolute_import, division
import time
import sys
import math
from api import vrep

import numpy as np
from actor_net import ActorNet
from critic_net import CriticNet
from actor_net_bn import ActorNet_bn
from critic_net_bn import CriticNet_bn
from collections import deque
import random
from tensorflow_grad_inverter import grad_inverter



class ACTION :
    #정의에서 env를 지웠다.
    def __init__(self, clientID = 0, opmode_blocking =0 ): 
        #스테이터스 개수
        #액션 개수
        self.num_states = 4
        self.num_actions = 4

        #steps 총 시간임
        self.steps = 10000

        #액션의 최대, 최솟값 배열
        self.high = [0.0, 0.0, 0.0, 0.0]
        self.low = [0.0, 0.0, 0.0, 0.0]

        #client ID, opmode_blocking
        self.clientID = clientID
        self.opmode_blocking = opmode_blocking

        # get joint handles:
        self.Motor_handle = [
            vrep.simxGetObjectHandle(clientID, 'Motor_left', opmode_blocking)[1],
            vrep.simxGetObjectHandle(clientID, 'Motor_right', opmode_blocking)[1]
        ]

        # get Steering handles :
        self.Steering_handle = [
            vrep.simxGetObjectHandle(clientID, 'steeringLeft', opmode_blocking)[1],
            vrep.simxGetObjectHandle(clientID, 'steeringRight', opmode_blocking)[1]
        ]

        # get target handle
        self.target_handle = vrep.simxGetObjectHandle(clientID, 'target', opmode_blocking)[1]

        # get Car handle
        self.car_handle = vrep.simxGetObjectHandle(clientID, 'Car', opmode_blocking)[1]


        #step을 실행할 때 액션의 개수에 따라서 아래의 액션 배열을 만들어야 한다.
        #action값을 넣으면 실행하는 함수들이 나오고, observation,reward,done,info 를 리턴한다.
        #status를 구하고, reward를 구하고, done을 구하고, info를 리턴해야 한다.
    def step (self, action = [0,0,0,0]) :

        vrep.simxSetJointTargetVelocity(self.clientID, self.Motor_handle[0], action[0], self.opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.clientID, self.Motor_handle[1], action[1], self.opmode_blocking)

        # desired pose of end-effector
        self.x_d, self.y_d, self.z_d = vrep.simxGetObjectPosition(self.clientID, self.car_handle, -1, self.opmode_blocking)[1]
        self.observation = [self.x_d, self.y_d, self.z_d, 0.0]
        self.reward = 0.0
        self.done = 0.0
        self.info = 0.0

        #step 줄어들게 하는 코드
        self.steps = self.steps-1

        return self.observation, self.reward, self.done, self.info
       
        #status 값들을 초기화 하는 코드다. 현재의 위치값을 리턴하면 된다.
    def reset(self) :
        self.observation = [0.0,0.0,0.0,0.0]
        return self.observation






   
        
                
        
        
        
                
        
        
        
                     
                 
        



