'''
让飞机以特定角度通过一个点
(5531,135,3400)
(5531,135,3406)
'''

import numpy as np
import os
import threading
import time
from rllab.spaces.discrete import Discrete
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.spaces.box import Box
import rllab.misc.logger as logger
from rllab.envs.socketManager import SocketSever
import rllab.misc.global_params as gp
import subprocess
import signal
import rllab.misc.global_params as gp
import math
BUFSIZE = 512
simulator_handle = None
#datafile2 = open(  './results/statefile.txt' , 'w')

def cart2sph(position):
	hxy = np.hypot(position[0],position[2])
	r = np.hypot(hxy,position[1])
	el = np.arctan2(position[1],hxy)
	az = np.arctan2(position[2],position[0])
	return np.array([az,el,r])


def run_simulator(ip,port):
    global simulator_handle
    
    cmd =["./FighterPilot",ip,str(port)]
    simulator_handle=subprocess.Popen(cmd,shell=False,close_fds= True,cwd="../../monitor/project/fighterpilot",stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def spherical_to_cartesian(r,theta,phi): #theta和phi弧度制
    x = r*math.sin(theta)*math.cos(phi)
    y = r*math.sin(theta)*math.sin(phi)
    z = r*math.cos(theta)
    return [x,y,z]


def cal_degree_of_two_vector(a,b):
    a_np_array = np.array(a)
    b_np_array = np.array(b)
    length_a = np.sqrt(a_np_array.dot(a_np_array))
    length_b = np.sqrt(b_np_array.dot(b_np_array))
    return np.nan_to_num(np.arccos(a_np_array.dot(b_np_array)/length_a/length_b)) 

def cal_cos_of_two_vector(a,b):
    a_np_array = np.array(a)
    b_np_array = np.array(b)
    length_a = np.sqrt(a_np_array.dot(a_np_array))
    length_b = np.sqrt(b_np_array.dot(b_np_array))
    return np.nan_to_num(a_np_array.dot(b_np_array)/length_a/length_b)

def cal_distance(a,b):
    cum = 0
    for i in range(len(a)):
        cum += np.square(a[i] - b[i])

    return np.sqrt(cum)


def cal_intersection_of_the_line_and_circle(x1,x2,x0,n):
    alpha = np.dot(x0-x1,n)/np.dot(x1-x2,n)
    return (np.array(x1)-np.array(x2))*alpha +np.array(x1)


class FighterpilotEnv(Env):
    def __init__(self,ip,port):
        self.action = ['V','a','a','a','a','a','a','a','V','a','a','a','a','a','a','a']
        self.ACTIONSIZE = 16
        self.MAXWAITNUM = 5
        self.state = None   #两个人的state都在其中
        self.recvStateCnt = 0
        self.trajection_cnt = 0
        self.total_time = 0
        self.distance = 0
        self.timeOut = 0
        self.missileRunOut = 0
        self.IP = ip
        self.PORT = port
        self.state = None
        self.old_state = None  #需要手动设置
        self.attitude = None
        self.goal = None  #np.array
        self.goals = []  #list 手动添加没有初始值
        #self.goals = [[0,0,0,0,0,0]]        #自动添加 有初始值
        self.radius_of_goal_circle = 0.02
        self.flag = False
        global socket 
        

    def extract_info(self,states):
#        print("test",states)
        
        splitIndex = states.find('@')
        strState1 = states[:splitIndex]
        strState2 = states[splitIndex+1 : len(states)-4]
#        print(strState1)
#        print(strState2)
        strState1Param = strState1.split()
        strState2Param = strState2.split()
#        print(strState1Param)
#        print(strState2Param)
#        exit()
        state1Param = [float(x) for x in strState1Param]
        state2Param = [float(x) for x in strState2Param]
        
        self.timeOut = int(states[len(states)-1])
        self.missileRunOut = int(states[len(states)-3])
        dead1 = int(state1Param[14])
        dead2 = int(state2Param[14])
        state1Param = state1Param + [0]*(40-len(state1Param))
        state2Param = state2Param + [0]*(40-len(state2Param))
        state = np.array(state1Param+state2Param)
        is_timeout_or_missileRunout = False
        is_dead = False
        if self.timeOut or self.missileRunOut:
            is_timeout_or_missileRunout = True  
        if dead1 or dead2:
            is_dead = True
        return np.copy(state1Param[0:3]), np.copy(state1Param[7:9]),is_timeout_or_missileRunout,is_dead
    
    @property
    def observation_space(self):
        return Box(-np.inf , np.inf , shape = (12,1))

    @property
    def action_space(self):
        return  Discrete(9)


    def reset(self):        
        #time1 = time.time()
        self.recvStateCnt = 0
        if not gp.sock_with_simulator.conn is None:
            gp.sock_with_simulator.conn.close()
        
        self.trajection_cnt += 1
        # print(self.trajection_cnt,'\n')
        global simulator_handle
        if not simulator_handle is  None:
             simulator_handle.kill()
   
        run_simulator(self.IP,self.PORT)    
        gp.sock_with_simulator.accept()
        
        statestr = gp.sock_with_simulator.conn.recv(BUFSIZE).decode()
#        self.total_time += time2-time1
        self.generate_goals()    #手动生成目标
        self.goal = self.get_goal()
        self.state,self.attitude,_ , _ = self.extract_info(statestr)
        print("=====================")
        print("goal:",self.goal)
#        print(self.state)
        observation = np.copy(self.state)
        goal_position = np.copy(self.goal[0:3])
        goal_direction = np.copy(self.goal[3:])
        height = np.copy(self.state[1])
        relative_coordinate = goal_position-observation
#        print(self.state)
        self.old_state = np.array([5531,135,3394])
        direction = (observation - self.old_state)/np.linalg.norm(observation - self.old_state)
#        print(direction)
        observation = np.hstack((relative_coordinate,goal_direction,self.attitude,direction,height))
       
#        print(observation)
        return observation


    def step(self , action):
        """
        0:Climb
        1:Dive
        2:Turn Left
        3:Turn Right
        4:accelarate
        """
        self.recvStateCnt += 1
        
        done = False
        #print("step")
        #print(done)
        #１5个动作　上下左右不转(加速)　　上下左右不转(减速)　　　上下左右不转(不加不减) 
        # if (action%5)<=3:
        #     self.action[action%5+1] = 'p'
        # do_vectory_change = int(action / 5)
        # if do_vectory_change == 0:
        #     self.action[5]='p'
        # if do_vectory_change == 1:
        #     self.action[6]='p'

        9 actions without accelerate
        if (action%3) <= 1:
            self.action[action%3+1] = 'p'
        if_spin = int(action / 3)
        if if_spin == 0:
            self.action[3]='p'
        if if_spin == 1:
            self.action[4]='p'
        #27 full actions
        # if_accelarate = int(action / 9)
        # if if_accelarate == 0:
        #     self.action[5]='p'
        # if if_accelarate == 1:
        #     self.action[6]='p'
        # spin_or_climb = action % 9
        # if (spin_or_climb%3 <= 1):
        #     self.action[action%3+1] ='p'
        # if_spin = int(spin_or_climb / 3)
        # if if_spin ==0:
        #     self.action[3] = 'p'
        # if if_spin ==1:
        #     self.action[4] = 'p'
        # 2^6
        # for i in range(0,6):
        #     if action[i]>0:
        #         self.action[i+1] ='p'
        # print(str(self.action)+'\n'+str(action))
        gp.sock_with_simulator.conn.sendall(''.join(self.action).encode(encoding='utf-8'))
#        print(''.join(self.action))
#        gp.datafile.write(''.join(self.action)+'\n')
#        datafile2.write('['+str(self.recvStateCnt)+']\n' + str(self.state[:40]) + '\n' + str(self.state[40:]) + '\n')
        self.action = ['V','a','a','a','a','a','a','a','V','a','a','a','a','a','a','a']
        #print(self.recvStateCnt)
        
        #statestr = sm.conn.recv(BUFSIZE).decode()
        statestr_b = gp.sock_with_simulator.conn.recv(BUFSIZE)      
        statestr = statestr_b.decode()
        self.state, self.attitude, is_timeout_or_missileRunout, is_dead = self.extract_info(statestr)
        
        print(self.state)      
#        if self.recvStateCnt == 50:           #自动添加到达状态为目标
#            self.add_goal(self.state)
        
        is_reach,reward = self.calculate_reward()
        print(reward)
        if is_timeout_or_missileRunout:
            done = True
       
        if is_dead:
            done = True
            reward = - 10
        
        if is_reach:
            print("reach   "+str(self.goal))
            # print(reward)
            done = True

        next_observation = np.copy(self.state)
        goal_position = np.copy(self.goal[0:3])
        goal_direction = np.copy(self.goal[3:])
        height = np.copy(self.state[1])
        relative_coordinate = goal_position-next_observation
        direction = (next_observation - self.old_state)/np.linalg.norm(next_observation - self.old_state)
        next_observation = np.hstack((relative_coordinate,goal_direction,self.attitude,direction,height))
        # print(next_observation)
        self.old_state = self.state
        return Step(observation = next_observation , reward = reward ,done = done)


    def render(self):
        pass

    def horizon(self):
        pass

    def calculate_reward(self):
#        distance = cal_distance(self.old_state,self.goal[0:3])
#        if distance > self.radius_of_goal_circle:
#            return 0
#         old_vector = self.old_state - self.goal[0:3]
#         new_vector = self.state - self.goal[0:3]
#         old_cos_theta = cal_cos_of_two_vector(old_vector,self.goal[3:])
# #        print("old_cos_theta：",old_cos_theta)
#         new_cos_theta = cal_cos_of_two_vector(new_vector,self.goal[3:])
# #        print("new_cos_theta：",new_cos_theta)
#         if old_cos_theta * new_cos_theta > 0 or (old_cos_theta == 0 and new_cos_theta == 0):
#             return 0
#         else:
#             cos_of_direction = cal_cos_of_two_vector((self.state - self.old_state),self.goal[3:])
#             intersection = cal_intersection_of_the_line_and_circle(self.old_state,self.state,self.goal[:3],self.goal[3:])
#             distance = cal_distance(intersection,self.goal[:3])
#             if distance >= self.radius_of_goal_circle:
#                 return 0
# #            print("intersection：",intersection.tolist())
#             reward = cos_of_direction*np.exp(-distance/5)
# #            print(reward)
#             return reward
        reach_flag = False
        forward_vector = self.state-self.old_state
        k = np.dot(self.goal[:3]-self.old_state,self.goal[3:]) / np.dot(forward_vector,self.goal[3:])
        Intersection = k*forward_vector + self.old_state
        distance = cal_distance(Intersection,self.goal[:3])
        k2 = k - 1
        if k < 0:
            reward = -1
        else:
            if distance >self.radius_of_goal_circle:
                reward = -1
            else:
                if k2*k > 0:
                    # reward = np.fabs(cal_cos_of_two_vector(self.goal[:3]-self.state,forward_vector))
                    #reward = (self.radius_of_goal_circle-distance)/self.radius_of_goal_circle
                    reward = -1
                else:
                    reach_flag = True
                    reward = cal_cos_of_two_vector(forward_vector,self.goal[3:])* np.exp(-distance/5)*10

        return reach_flag,reward


    def generate_goals(self):
#        goal_radius  = 6
        init_x = 5531.0
        init_y = 135.0
        init_z = 3400.0
        init_coordinate = np.array([init_x,init_y,init_z])
##        for i in range(init_y-goal_radius,init_y+goal_radius+self.radius_of_goal_circle,self.radius_of_goal_circle*2):
##            x = init_x
##            y = i
##            z = init_z+160
##            goal_coordinate = np.array([x,y,z])
##            direction = (goal_coordinate - init_coordinate)/np.linalg.norm(goal_coordinate - init_coordinate)
##            self.goals.append(np.hstack((goal_coordinate,direction)))
#        goal_coordinate = np.array([5531,130,3560])  
#        direction = (goal_coordinate - init_coordinate)/np.linalg.norm(goal_coordinate - init_coordinate)
#        self.goals.append(np.hstack((goal_coordinate,direction)))  
        goal_x = 5531.0
        goal_y = 135.0
        goal_z = 3413.33
        y = 134.94
        for i in range(0,3):         
            x = goal_x
            y = y+self.radius_of_goal_circle
            z = goal_z
            goal_coordinate = np.array([x,y,z])
            direction = (goal_coordinate - init_coordinate)/np.linalg.norm(goal_coordinate - init_coordinate)
            y = y + self.radius_of_goal_circle
            self.goals.append(np.hstack((goal_coordinate,direction)))
    def add_goal(self,state):   #添加一个
        if self.flag == True :
            return 
        init_x = 5531
        init_y = 135
        init_z = 3400
        init_coordinate = np.array([init_x,init_y,init_z])
#        print("==========",type(self.goals[0]))
#        exit()
#        if (len(self.goals) == 1) and np.array_equal(self.goals[0],[0,0,0,0,0,0]):
        if self.flag == False:   
            self.goals.clear()
            self.flag =True
        direction = (state - init_coordinate)/np.linalg.norm(state - init_coordinate)
        self.goals.append(np.hstack((state,direction)))
#        for i in range(-1,2):
#            x = init_x
#            y = init_y + i*2*self.radius_of_goal_circle
#            z = init_z+160
#            goal_coordinate = np.array([x,y,z])
#            direction = (goal_coordinate - init_coordinate)/np.linalg.norm(goal_coordinate - init_coordinate)
#            self.goals.append(np.hstack((goal_coordinate,direction)))   
    def get_goal(self):
        index = np.random.randint(0,len(self.goals))
        return np.array(self.goals[index])


if __name__ == '__main__':
    env = FighterpilotEnv("127.0.0.1",10000)
    env.generate_goals()
    print(env.goals)
