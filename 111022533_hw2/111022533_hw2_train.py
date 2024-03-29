from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import numpy as np  
import math 
from random import shuffle
import tensorflow.keras.backend as K
import tensorflow as tf
from copy import deepcopy

import os

# from matplotlib import pyplot as plt
from PIL import Image

# os.environ['CUDA_VISIBLE_DEVICES']=''

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
 
# # actions for more complex movement
# COMPLEX_MOVEMENT = [
#     ['NOOP'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
#     ['left', 'A'],
#     ['left', 'B'],
#     ['left', 'A', 'B'],
#     ['down'],
#     ['up'],
# ]

class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]

'''
 state.shape: (240, 256, 3), 
 reward: 0.0, 
 info: {'coins': 0, 'flag_get': False, 'life': 2, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40, 'y_pos': 79}
'''

 
class Agent:

    def __init__(self, name, para):   
        self.para = para 
        self.model = self.build_model(name)
     
        
 
    def build_model(self, name):
        # input: state
        # output: each action's Q-value
        input_shape = [self.para.img_shape[0], self.para.img_shape[1], self.para.img_stack_num]
        screen_stack = tf.keras.Input(shape=input_shape, dtype=tf.float32)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4)(screen_stack) # (4, 8, 8, 32)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2)(x) # (32, 4, 4, 64)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1)(x) # (64, 3, 3, 64)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=512)(x)
        x = tf.keras.layers.ReLU()(x)
        Q = tf.keras.layers.Dense(self.para.action_num)(x)

        model = tf.keras.Model(name=name, inputs=screen_stack, outputs=Q)

        return model

    def compute_loss(self, state, action, reward, tar_Q, notTernimal):
        output = self.model(state)
        index = tf.stack([tf.range(tf.shape(action)[0]), action], axis=1)
        Q = tf.gather_nd(output, index)
        # tar_Q *= ~np.array(ternimal)
        tar_Q *= notTernimal
        loss = tf.reduce_mean(tf.square(reward + self.para.discount_factor * tar_Q - Q))
        return loss

    def max_Q(self, state):
        output = self.model(state)
        return tf.reduce_max(output, axis=1)

    def select_action(self, state): 
        if np.random.rand() < self.para.eps_cur:
            action = np.random.choice(self.para.action_num)   
        else:
            state = np.expand_dims(state, axis = 0)
            output = self.model(state)
            action = tf.argmax(output, axis=1)[0]
            action = int(action.numpy())

        return action


    def update_parameters(self, i):
        para = self.para  
        para.eps_cur = para.eps_begin + (i / para.iter_num) * (para.eps_end - para.eps_begin)


    # def shutdown_explore(self):
    #     # make action selection greedy
    #     self.para.exploring_rate = 0
 

    def save_checkpoint(self, path):  
        print(f'- saved ckpt {path}') 
        self.model.save_weights(path)
         
    def load_checkpoint(self, path): 
        # need call once to enable load weights.
        print(f'- loaded ckpt {path}') 
        self.model(tf.random.uniform(shape=[1, self.para.img_shape[0], self.para.img_shape[1], 
                                                        self.para.img_stack_num]))
        self.model.load_weights(path)




class Replay_buffer():

    def __init__(self, buffer_size=50000):
        self.experiences = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.experiences) >= self.buffer_size:
            self.experiences.pop(0)
        self.experiences.append(experience)

    def sample(self, size):
     
        if size > len(self.experiences):
            idx = np.random.choice(len(self.experiences), size=size)
        else:
            idx = np.random.choice(len(self.experiences), size=size, replace=False)

        # from all sampled experiences, extract a tuple of (s,a,r,s')
        states = []
        actions = []
        rewards = []
        states_prime = []
        terminal = []
        for i in range(size):
            exp = self.experiences[idx[i]]
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            states_prime.append(exp[3])
            terminal.append(exp[4])

        return states, actions, rewards, states_prime, terminal



 

def rgb2gray(rgb):  
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def preprocess_screen(screen): 
    screen = screen[::2,::2,:]
    screen = rgb2gray(screen) 
    screen = screen[..., np.newaxis] # shape is (h, w, 1)
    return screen

def stack_frames(input_frames):
    if(len(input_frames) == 1):
        state = np.concatenate(input_frames*4, axis=-1)
    elif(len(input_frames) == 2):
        state = np.concatenate(input_frames[0:1]*2 + input_frames[1:]*2, axis=-1)
    elif(len(input_frames) == 3):
        state = np.concatenate(input_frames + input_frames[2:], axis=-1)
    else:
        state = np.concatenate(input_frames[-4:], axis=-1)
    return state
 

class Trainer():

    def __init__(self, para, buffer):

        self.para = para
        self.buffer = buffer 
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        
        self.online_agent = Agent('online', para) 
        if('ckpt_load_path' in self.para): 
            self.online_agent.load_checkpoint(self.para.ckpt_load_path)
         

        self.target_agent = Agent('target', para) 
        self.target_agent.model.set_weights(self.online_agent.model.get_weights())
 

    def executeEpisode(self, explore=True, saveToBuffer=True):

        state = env.reset()       
        input_frames = [preprocess_screen(state)]
        state_stack = stack_frames(input_frames)

        cum_reward = 0
        logs = []

        t = 0
        done = False
        while not done:

            if t % 4 == 0:
                action = self.online_agent.select_action(state_stack)

            state_next, reward, done, info = env.step(action)                
            cum_reward += reward
            
            if(len(input_frames)>=4): input_frames.pop(0)
            input_frames.append(preprocess_screen(state_next))
            state_next_stack = stack_frames(input_frames)  # get next state
 
            # if reward > 0 or reward < -5 or np.random.rand() < 0.05: 
            if t % 4 == 0:
                self.buffer.add((state_stack, action, reward, state_next_stack, done)) 
                
                log = {}
                log['t'] = t 
                log['done'] = done 
                log['reward'] = reward                                                
                for i in ['life','score','time','x_pos', 'y_pos']: log[i] = info[i]
                logs.append(log) 
            
            state_stack = state_next_stack
            t += 1

        return cum_reward, logs


    def train(self):

        cum_rewards = []

        for i in range(self.para.iter_num):
            print(f'##########################')
            print(f'iter: {i}')
            

            with open("log.txt", "w") as f: 
                f.write(f"iter: {i}\n")
                    
            # for episode in range(1, self.para.episode_num + 1):
            cum_reward, logs = self.executeEpisode()  
            with open("log.txt", "a") as f: 
                for log in logs:
                    f.write(str(log)+'\n')
            
            cum_rewards.append(cum_reward)
            print(f'cum_rewards: {cum_rewards}')
            print(f'len(self.buffer.experiences): {len(self.buffer.experiences)}')

            
            n = int(len(self.buffer.experiences) / self.para.batch_size)
            for epoch in range(self.para.epoch_num):  
                losses = [] 
                for j in range(n):
                    experience = self.buffer.sample(self.para.batch_size)
                    loss = self.train_step(experience)
                    losses.append(loss)
                print(f'epoch: {epoch}, loss: {np.mean(losses)}')

            if i % self.para.update_target_agent_period == 0:
                self.target_agent.model.set_weights(self.online_agent.model.get_weights())

            if i % self.para.save_period == 0:
                self.online_agent.save_checkpoint(self.para.ckpt_save_path)


            self.online_agent.update_parameters(i)
            print(f'self.para.eps_cur: {self.para.eps_cur}')



    def train_step(self, experience):

        # states: a list of 4 stacked imgs of shape (h, w, 4).
        states, actions, rewards, states_next, terminal = experience      
        
        states = np.asarray(states).reshape(-1, self.para.img_shape[0], self.para.img_shape[1], self.para.img_stack_num)
        states_next = np.asarray(states_next).reshape(-1, self.para.img_shape[0], self.para.img_shape[1], self.para.img_stack_num)
    
        states = tf.convert_to_tensor(states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.int32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        states_next = tf.convert_to_tensor(states_next, tf.float32)
        # terminals = tf.convert_to_tensor(terminal, tf.bool)
        
        # print(f'terminal:{terminal}')
        notTerminals = tf.convert_to_tensor(~np.array(terminal), tf.float32)

        # print(f'notTerminals: {notTerminals}')

        loss = self._train_step(states, actions, rewards, states_next, notTerminals)
        return loss.numpy()


    @tf.function
    def _train_step(self, state, action, reward, next_state, notTerminal):
        
        tar_Q = self.target_agent.max_Q(next_state)

        with tf.GradientTape() as tape:
            loss = self.online_agent.compute_loss(state, action, reward, tar_Q, notTerminal)

        gradients = tape.gradient(loss, self.online_agent.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_agent.model.trainable_variables))
    
        return loss


    # def evaluate(self):






para = AttrDict({
    'action_num': len(COMPLEX_MOVEMENT), 
    # 'img_shape': (240, 256, 3),
    'img_shape': (120, 128, 3),
    'img_stack_num': 4,

    'buf_size': 2**12,
    
    'iter_num': 100000,
    'episode_num': 1,
    'epoch_num': 25,
    'batch_size': 32,
    'update_target_agent_period': 1,
    'save_period': 5,
 

    'discount_factor': 0.99,
    
    'eps_cur': 0.1,
    'eps_begin': 0.1,
    'eps_end': 0.01,
    

    'ckpt_save_path': "111022533_hw2/ckpt/checkpoint2.h5",
    'ckpt_load_path': "111022533_hw2/ckpt/checkpoint1.h5"
})



buffer = Replay_buffer(para.buf_size)
trainer = Trainer(para, buffer)
trainer.train()




