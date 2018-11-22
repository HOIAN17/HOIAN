from __future__ import print_function, absolute_import, division
import numpy as np
from ddpg2 import DDPG
from ou_noise import OUNoise
from action import ACTION
import time
import sys
import math
from api import vrep

#specify parameters here:
episodes= 10000
is_batch_norm = False #batch normalization switch

from time import time as t
def xrange(x) :

    return iter(range(x))

def main(): 
    try:
        vrep.simxFinish(-1) # just in case, close all opened connections
    except:
        pass

    opmode_blocking = vrep.simx_opmode_blocking

    clientID=vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP

    if clientID != -1:
        print('Connected to remote API server')

        # enable the synchronous mode on the client:
        vrep.simxSynchronous(clientID, True)

        # start the simulation:
        vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(clientID)

        # action 객체를 만든다.
        action = ACTION()
        steps = action.steps

        #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
        agent = DDPG(action , is_batch_norm)
        #env.acton_space.shape[0]은 액션 입력값의 개수처럼 보인다. 액션 입력값 개수로 바꾸겠다. 좌, 우, 속도 증가, 속도 감소
        exploration_noise = OUNoise(agent.num_actions)
        counter=0
        reward_per_episode = 0    
        total_reward=0
        #env.opservation_space.shape[0]은 state의 개수로 보인다. 이 개수는 아직 미정인데 현재 생각할 수 있는 개수는 1. 도로의 중앙선으로부터의 거리,
        #2. 타겟까지의 거리, 3. 장애물과의 거리
        num_states = agent.num_states
        #env.acton_space.shape[0]은 액션 입력값의 개수처럼 보인다. 액션 입력값 개수로 바꾸겠다. 좌, 우, 속도 증가, 속도 감소
        num_actions = agent.num_actions 
        print ("Number of States:", num_states)
        print ("Number of Actions:", num_actions)
        print ("Number of Steps per episode:", steps)
        #saving reward:
        reward_st = np.array([0])
      
    
        for i in xrange(episodes):
            print ("==== Starting episode no:",i,"====","\n")
            #observation에는 status 값들이 들어간다. 배열로 되어 있고 행 열은 모르겠다.. 초기화 하는 단계
            observation = action.reset()
            reward_per_episode = 0
            for t in xrange(steps):
                #rendering environmet (optional) 환경을 초기화함            
                #env.render()
                x = observation
                #X 라는 observation 배열을 1, 스테이터스 개수 형태로 바꾼다. 그러니 아마 스테이터스 값들은 1행의 값들의 배열로 저장하면 되지 않을까
                #agent를 사용하는데, 이 함수의 변형 필요.. env 가 뭔지 모르겠.. DDPG 파일에 생성자에서 정의에 대입하도록 되어 있는데 이걸 바꿔서 사용해야함.
                #env는 일종의 객체고 이 안에 스테이터스, 액션 개수 값들이 들어있음. 원 코드를 확인해야 함
                #time step마다 observation값들을 대입할 수 있도록 그걸 구하는 코드를 넣거나, 혹은 클래스를 만들어서 사용하면 된다. 새로운 클래스 만들면서 넣어도 됨
                action_value = agent.evaluate_actor(np.reshape(x,[1,num_states]))
                noise = exploration_noise.noise()
                action_value = action_value[0] + noise #Select action according to current policy and exploration noisealizing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                
                print ("Action at step", t ," :",action_value,"\n")
            
                #env.step 을 대신할 수 있는 함수가 필요함!! 원 env 코드를 변형시켜서 사용하면 될거 같음 아니면 env를 대체할 수 있는 다른 파일 만들기
                observation,reward,done,info = action.step(action_value)
                print(observation)
            
                #add s_t,s_t+1,action,reward to experience memory 이전의 스테이터스 값, 액션 실행 후 스테이터스 값, 현재 입력한 액션값, 리워드 값, 종료 여부
                agent.add_experience(x,observation,action_value,reward,done)
            
                #train critic and actor network
                if counter > 64: 
                    agent.train()
                
                reward_per_episode+=reward
                counter+=1
                
                #check if episode ends:
                if (done or (t == steps-1)):
                    print ('EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode)
                    print ("Printing reward to file")
                    exploration_noise.reset() #reiniti.txt',reward_st, newline="\n")
                    reward_st = np.append(reward_st,reward_per_episode)
                    np.savetxt('episode_reward.txt',reward_st, newline="\n")
                    print ('\n\n')
                    break
            total_reward+=reward_per_episode            
            print ("Average reward per episode {}".format(total_reward / episodes)  )
    else :
        print ('Failed connecting to remote API server')
    print ('Program ended')



if __name__ == '__main__':
    main()  