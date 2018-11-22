from gym import error, spaces
from api import vrep
from time import sleep
import numpy as np

class VrepEnv:
    '''
    '''
    def __init__(self, enable_acturator_dynamics = False):
        # Setting for DDPG
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,))
        
        self.action_space = spaces.Box(low=-0.,high=1., shape=(2,))
        
        self.vrep_client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) #connect to V-REP
        
        while True:
            print('Connected to remote API server')
            if self.vrep_client_id != -1:
                break
            else:
                sleep(0.2)
                
        self.motor_handles = [
            vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='Motor_left', operationMode=vrep.simx_opmode_blocking)[1],
            vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='Motor_right', operationMode=vrep.simx_opmode_blocking)[1]
        ]

        self.goal_handle = vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='Goal', operationMode=vrep.simx_opmode_blocking)[1]
        self.robot_handle = vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='Robot', operationMode=vrep.simx_opmode_blocking)[1]

    def getGoalDistance(self):
        abs_goal_position = vrep.simxGetObjectPosition(clientID=self.vrep_client_id,
                                                    objectHandle=self.goal_handle,
                                                    relativeToObjectHandle=-1,
                                                    operationMode=vrep.simx_opmode_blocking)[1]
        
        abs_goal_position = np.asarray(abs_goal_position)
    
        abs_robot_position = vrep.simxGetObjectPosition(clientID=self.vrep_client_id,
                                                   objectHandle=self.robot_handle,
                                                   relativeToObjectHandle=-1,
                                                   operationMode=vrep.simx_opmode_blocking)[1]
        abs_robot_position = np.asarray(abs_robot_position)
    
        distance = np.linalg.norm(abs_goal_position-abs_robot_position)
    
        # print("Robot Position", abs_robot_position, "abs_goal_position", abs_goal_position)
        # print("Distance", distance)
    
        return distance

    def getLidarData(self):
        _, lidar_data_bin = vrep.simxGetStringSignal(clientID=self.vrep.client_id, signalName='LiDAR', operationMode=vrep.simx_opmode_blocking)
        lidar_data = np.array(vrep.simxUnpackFloats(lidar_data_bin), dtype=float)

        return lidar_data

    def getMinLidarData(self):
        try:
            return np.min(self.getLidarData())
        except:
            return self.min_lidar_data

    def setMotorSpeeds(self, action):

            vrep.simxSetJointTargetVelocity(clientID=self.vrep_client_id,
                                            jointHandle=self.motor_handles[0],
                                            targetVelocity=action[0],
                                            operationMode=vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(clientID=self.vrep_client_id,
                                             jointHandle=self.motor_handles[1],
                                            targetVelocity=action[1],
                                            operationMode=vrep.simx_opmode_blocking)

    def getReward(self):
            return (self.prev_goal_distance - self.goal_distance)*100 - 2 * np.exp(-(self.min_lidar_data-0.08)**2/(2*0.05**2))

    def step(self,action):






        self.setMotorSpeeds(action)

        vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)

        self.goal_distance = self.getGoalDistance()
        self.min_lidar_data = self.getMinLidarData()

        obs = [self.goal_distance, self.min_lidar_data, action[0], action[1]]

        reward = self.getReward()

        print(self.prev_goal_distance, obs, reward)
        self.prev_goal_distance = self.goal_distance
        return obs, reward

    def reset(self):



        print("ENV reset")
        r = vrep.simxStopSimulation(clientID=self.vrep_client_id, operationMode=vrep.simx_opmode_blocking)
        print("sin stopped", r==0)
        sleep(0.2)
        r = vrep.simxStartSimulation(clientID=self.vrep_client_id, operationMode=vrep.simx_opmode_blocking)
        print("sim started", r==0)

        vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)

        self.prev_goal_distance = 0
        self.goal_distance = self.getGoalDistance()
        self.min_lidar_data = self.getMinLidarData()


        return [self.goal_distance, self.min_lidar_data, 0, 0]


    def finishCheck(self, obs):
            d = obs[0]
            l = obs[1]

            if d< 0.05:
                return True
            elif l < 0.08:
                return True
            elif d > 1.5:
                return True
            else:
                return False