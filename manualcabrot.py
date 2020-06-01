# Importing the libraries
import numpy as np
import os
import time
import torch
import math 

import os
import pygame
from math import sin, radians, degrees, copysign
from pygame.math import Vector2


from PIL import Image as PILImage
from models import ReplayBuffer, TD3
from scipy.ndimage import rotate


seed = 0  # Random seed number
# Set seed for consistency
torch.manual_seed(seed)
np.random.seed(seed)
save_models = True  # Boolean checker whether or not to save the pre-trained model

file_name = "%s_%s_%s" % ("TD3", "CarApp", str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")
if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
last_reward = 0
origin_x = 743
origin_y = 302
scores = []
#im = CoreImage("./images/MASK1.png")

# Some possible locations for pickup/drop points
# coordinates = [[110,270],[200,280],[300,560],[320,475],[590, 275],\
# [570,340],[635,580],[900, 380],[1065,470],[1100, 305],[1330,380],[1220,617],[1080,190]]
#coordinates = [[1330,380],[1220,617],[1080,190]]
coordinates = [[590,318],[756,126]]
first_update = True # Setting the first update
last_distance = 0   # Initializing the last distance

def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global sandCount
    sand = np.zeros((longueur, largeur))
    img = PILImage.open("mask.png").convert('L')
    sand = np.asarray(img) / 255
    goal_x, goal_y = coordinates[np.random.randint(0, 2)]
    # goal_x = 1220
    # goal_y = 622
    first_update = False
sandCount=0
global gView
class Car:
    cropsize = 28
    padsize = 28
    rotation = 0
    view = np.zeros([1,int(cropsize),int(cropsize)])
    
    def __init__(self, x=743, y=302, angle=0.0, length=4, max_steering=30, max_acceleration=5.0):
        self.position = Vector2(743,302)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 20
        self.brake_deceleration = 10
        self.free_deceleration = 2
        global sandCount
        self.acceleration = 0.0
        self.steering = 0.0
        # attributes of car 
        #self.pos = [743,302]
        self.x =0
        self.y =0
        self.center = [234,245]
        self.velocity = [0,0]
        self.view = np.zeros([1,int(28),int(28)])
        #self.car = {'x':0,'y':0,'center':[234,245],'velocity':[0,0],'view':tc.view}
    # def update(self, dt):
    #     self.velocity += (self.acceleration * dt, 0)
    #     #print(self.velocity)
    #     self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))

    #     if self.steering:
    #         turning_radius = self.length / sin(radians(self.steering))
    #         angular_velocity = self.velocity.x / turning_radius
    #     else:
    #         angular_velocity = 0

    #     self.position += self.velocity.rotate(-self.angle) * dt
    #     self.angle += degrees(angular_velocity) * dt
    def move(self, rotation):
        global episode_num
        global padsize
        global cropsize
        self.position = Vector2(*self.velocity) + self.position
        self.rotation = rotation
        self.angle = self.angle + self.rotation

        # Preparing the image for the state
        tempSand = np.copy(sand)
        tempSand = np.pad(tempSand,self.padsize,constant_values=1.0)
        tempSand = tempSand[int(self.position.x) - self.cropsize + self.padsize:int(self.position.x) + self.cropsize + self.padsize,
                   int(self.position.y) - self.cropsize + self.padsize:int(self.position.y) + self.cropsize + self.padsize]
        tempSand = rotate(tempSand, angle=90-(self.angle-90), reshape= False, order=1, mode='constant',  cval=1.0)
        tempSand[int(self.padsize)-5:int(self.padsize), int(self.padsize) - 2:int(self.padsize) + 3 ] = 0.6
        tempSand[int(self.padsize):int(self.padsize) + 5, int(self.padsize) - 2:int(self.padsize) + 3] = 0.3
        # tempSand = rotate(tempSand, angle= self.angle-90, order = 1, reshape = False, mode = 'constant', cval = 1.0)
        self.view=tempSand

        #To check if cropped image is fine
        if total_timesteps %50==10:
            # print("image", self.angle)
            img = PILImage.fromarray(np.uint8(self.view * 255) , 'L')
            imgname="1.png"
            img.save(imgname)
            img.show()
            time.sleep(2)
        self.view = self.view[::2, ::2]
        self.view = np.expand_dims(self.view, 0)
        gView = self.view

# reward = 0
# total_timesteps = 0
class Game:
    global sandCount
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 1429
        height = 660
        self.sWidth=1429
        self.sHeight=660
        #tc=Car()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.center = [234,245]
        self.car = Car()
        self.bg_img = pygame.image.load('MASK1.png')
        self.car_img = pygame.image.load('car.png')
        self.car_img = pygame.transform.scale(self.car_img,(40,20))
        #self.car = {'x':0,'y':0,'center':[234,245],'velocity':[0,0],'view':tc.view}
        #global reward
        global total_timesteps
        self.pickup = {"x":0,"y":0,"size":[]}
        self.drop ={"x":0,"y":0,"size":[]}
    

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector2(6, 0)


    def runCarUi(self,carposx,carposy,desposx,desposy):
        #self.screen.fill((255,255,255))
        self.screen.blit(self.bg_img, self.bg_img.get_rect())
        self.screen.blit(self.car_img,self.car.position) 
        pygame.draw.rect(self.screen, (0, 255, 0), (desposx, desposy, 25,25)) 
        pygame.display.flip()

    def reset(self):
        global last_distance
        global origin_x
        global origin_y
        self.car.position.x = origin_x
        self.car.position.y  = origin_y
        xx = goal_x - self.car.position.x
        yy = goal_y - self.car.position.y
        tempAngle = -(180 / math.pi) * math.atan2(
            self.car.velocity[0] * yy- self.car.velocity[1] * xx,
            self.car.velocity[0]* xx + self.car.velocity[1] * yy)
        orientation = tempAngle/180
        #orientation = Vector2(*self.car.velocity).angle((xx,yy))/180
        self.distance = np.sqrt((self.car.position.x- goal_x) ** 2 + (self.car.position.y - goal_y) ** 2)
        
        # state = [self.car.view, orientation, -orientation]
        # state = [self.car.view, orientation, -orientation, self.distance]
        #print(self.car)
        state = [self.car.view , orientation, -orientation, last_distance - self.distance]
        return state


    def step(self,action):
        #print(action)
        global goal_x
        global goal_y
        global origin_x
        global origin_y
        global done
        global last_distance
        global sandCount
        global distance_travelled

        rotation = action.item()
        print(rotation)
        #print("act-",action.item())
        self.car.move(rotation)
        #self.car.x = self.car.position.x
        #self.car.y = self.car.position.y
        self.runCarUi(self.car.position.x,self.car.position.y,goal_x,goal_y)
        #print(self.car.x , goal_x,self.car.y , goal_y)
        self.distance = np.sqrt((self.car.position.x - goal_x) ** 2 + (self.car.position.y - goal_y) ** 2)
        xx = goal_x - self.car.position.x
        yy = goal_y - self.car.position.y
        tempAngle = -(180 / math.pi) * math.atan2(
            self.car.velocity[0] * yy- self.car.velocity[1] * xx,
            self.car.velocity[0]* xx + self.car.velocity[1] * yy)
        orientation = tempAngle / 180.
        # state = [self.car.view, orientation, -orientation]
        # state = [self.car.view, orientation, -orientation, self.distance]
        state = [self.car.view, orientation, -orientation, last_distance-self.distance]
        #print(orientation, -orientation, last_distance,self.distance)
        # moving on the sand
        if sand[int(self.car.position.x), int(self.car.position.y)] > 0:
            self.car.velocity = Vector2(0.5, 0).rotate(self.car.angle)
            last_reward = -5.0 #-1
            # sandCount+=1
            # distance_travelled-=1

        else:  # moving on the road
            self.car.velocity = Vector2(1.5, 0).rotate(self.car.angle)
            last_reward = -1.5 #-0.2 # -2.0
            # distance_travelled+=0.5
            # if sandCount>0:
            #     sandCount-=1

            # moving towards the goal
            if self.distance < last_distance:
                last_reward = 0.5 #0.1
                # distance_travelled=+1
                # else:
                #     last_reward = last_reward +(-0.2)

        # Near the boundary
        if self.car.position.x < 5:
        
            self.car.position.x = 5
            last_reward = -10 #-1
            sandCount += 1
        if self.car.position.x > self.width - 5:
            self.car.position.x= self.width - 5
            last_reward = -10    #-1
            sandCount += 1
        if self.car.position.y < 5:
            self.car.position.y = 5
            last_reward = -10   #-1
            sandCount += 1
        if self.car.position.y > self.height - 5:
            self.car.position.y = self.height - 5
            last_reward = -10    #-1
            sandCount += 1

        # Reached goal
        if self.distance < 30:
            if self.topickup == 1:
                origin_x = goal_x
                origin_y = goal_y
                goal_x,goal_y= coordinates[np.random.randint(0,2)]
                # goal_x = 200
                # goal_y = 100
                self.drop['x'] = goal_x
                self.drop['y'] = goal_y
                self.topickup = 0
                last_reward = 100
                self.pickup['size'] = 20,20
                self.drop['size'] = 60, 40
                done = True
            else:
                origin_x = goal_x
                origin_y = goal_y
                # goal_x = 1220
                # goal_y = 622
                goal_x,goal_y= coordinates[np.random.randint(0, 2)]
                self.pickup['x'] = goal_x
                self.pickup['y'] = goal_y
                self.topickup = 1
                last_reward = 100
                self.pickup['size'] = 40, 40
                self.drop['size'] = 1, 1
                done = True

        # Carry the passenger to the drop location
        if self.topickup == 0:
            self.pickup['x'] = self.car.position.x - 5
            self.pickup['y'] = self.car.position.y - 5
        # if sandCount>100:
            # done=True
            # last_reward=distance_travelled
            # distance_travelled=0
        last_distance = self.distance
        return state, last_reward, done

    def evaluate_policy(self, policy, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = self.reset() # ToDo reset env
            done = False
            while not done:
                action = policy.select_action(np.array(obs))
                obs,reward,done = self.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print("---------------------------------------")
        print("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print("---------------------------------------")
        return avg_reward



    def update(self, dt):
        global scores
        global first_update
        global goal_x
        global goal_y
        global longueur
        global largeur
        global last_reward
        global reward
        global policy
        global done
        global episode_reward
        global replay_buffer
        global obs
        global new_obs
        global evaluations

        global episode_num
        global total_timesteps
        global timesteps_since_eval
        global episode_num
        global max_timesteps
        global max_episode_steps
        global episode_timesteps
        global distance_travelled
        self.width = 1429
        self.height = 660
        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            self.topickup=1
            self.pickup['x'] = goal_x
            self.pickup['y'] = goal_y
            evaluations = [self.evaluate_policy(policy)]
            distance_travelled=0
            done = True
            obs = self.reset()
        print("episoderew - ",episode_reward)
        if episode_reward<-2000:
            done=True
        
        print(total_timesteps,max_timesteps)
        if total_timesteps < max_timesteps:


            # If the episode is done
            if done:
                print("done-rwached")
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num,
                                                                                  episode_reward))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip,
                                 policy_freq)

                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= eval_freq:
                    print("reached-")
                    timesteps_since_eval %= eval_freq
                    evaluations.append(self.evaluate_policy(policy))
                    policy.save(file_name, directory="./pytorch_models")
                    np.save("./results/%s" % (file_name), evaluations)

                # When the training step is done, we reset the state of the environment
                obs = self.reset()

                # Set the Done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Before 10000 timesteps, we play random actions
            if total_timesteps < start_timesteps:
                action = np.random.uniform(low=-5, high=5, size=(1,))
            else:  # After start_timesteps, we switch to the model
                action = policy.select_action(np.array(obs))
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, size=1)).clip(
                        -5, 5)

            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs,reward, done = self.step(action)

            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(
                done)

            # We increase the total reward
            episode_reward += reward
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action, reward, done_bool))
            if total_timesteps%10==1:
                print(" ".join([str(total_timesteps), str(obs[1:]), str(new_obs[1:]), str(action), str(reward), str(done_bool)]))
            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
            # Saving model at every 5000 iterations
            if total_timesteps%5000==1:
                print("Saving Model %s" % (file_name))
                policy.save("%s" % (file_name), directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)
        else:
            action = policy.select_action(np.array(obs))
            new_obs,reward, done = self.step(action)
            obs = new_obs
            total_timesteps += 1
            if total_timesteps%1000==1:
                print(total_timesteps)


# Initializing Global Variables
start_timesteps = 3e3  # 1e4 Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 1e3  #5e3 How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e4  #5e5 Total number of iterations/timesteps

expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100  # Size of the batch
discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005  # Target network update rate
policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2  #
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0

episode_reward=0
t0 = time.time()
distance_travelled=0
max_episode_steps = 1000
done = True # Episode over
load_model=True # Inference. Set to false for training from scratch

state_dim = 4
action_dim = 1
max_action = 5

replay_buffer = ReplayBuffer()

policy = TD3(state_dim, action_dim, max_action)

obs=np.array([])
new_obs=np.array([])
evaluations=[]

if load_model == True:
    print("------------load model")
    total_timesteps = max_timesteps
    policy.load("%s" % (file_name), directory="./pytorch_models")

        

parent = Game()
parent.serve_car()
start_ticks=pygame.time.get_ticks()

#parent.run()
while True:
    
    parent.update(1/60)
    time.sleep(1/60) 

