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
import time
# from PIL import Image

# os.environ['CUDA_VISIBLE_DEVICES']=''


 

class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]
 

para = AttrDict({
    'action_num': len(COMPLEX_MOVEMENT), 
    'img_shape': (120, 128, 3), 
    'k': 4,
    'frame_shape': (120, 128, 1),
 
    
    'step_num': 1000000,
    'discount_factor': 0.99,
    
    'eps_begin': 1.0,
    'eps_end': 0.1,
    # 'eps_begin': 0.5,
    # 'eps_end': 0.1, 

    'buf_size': 450000, 

    'batch_size': 32,
    'lr': 2.5e-4,
       

    'replay_start_size': 10000,
    # 'replay_start_size': 100,

    'learning_period': 4,
    'target_update_period': 10000,
    'save_period': 10000, 
    'log_period': 250,
    'eval_period': 5000,
    # 'eval_period': 250,


    'ckpt_save_path': "111022533_hw2/ckpt/checkpoint0.h5",
    # 'ckpt_load_path': "111022533_hw2/ckpt/checkpoint0.h5"
})




# para = AttrDict({
#     'action_num': len(COMPLEX_MOVEMENT), 
#     'img_shape': (120, 128, 3),
#     'k': 4,
#     'frame_shape': (120, 128, 1),
 
    
#     'step_num': 1000000,
#     'discount_factor': 0.99,
#     'eps_begin': 1.0,
#     'eps_end': 0.1, 
#     'buf_size': 5000, 

#     'batch_size': 32,
#     'lr': 2.5e-4,
       

#     'replay_start_size': 200,
#     'learning_period': 4,
#     'target_update_period': 5,
#     'save_period': 50, 
#     'log_period': 1,


#     'ckpt_save_path': "111022533_hw2/ckpt/checkpoint3.h5",
#     # 'ckpt_load_path': "111022533_hw2/ckpt/checkpoint2.h5"
# })


 
class EnvWrapper:

    def __init__(self, env):   
        self.env = env 
        self.skip = 4

    def step(self, action):
        """
        - 4 steps at a time. 
        - take last obs, done and info, sum all 4 rewards.
        - clip reward between -1, 1.
        - return if encounter done before 4 steps.
        """        
        cum_reward = 0

        for i in range(self.skip):
        
            obs_next, reward, done, info = env.step(action)  
            cum_reward += reward

            if done: break

        # cum_reward = min(max(cum_reward, -1), 1)
        return obs_next, cum_reward, done, info
 
    def reset(self):
        return self.env.reset()



env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
 
env_wrapper =  EnvWrapper(env)



class Agent:

    def __init__(self, name, para):   
        self.para = para 
        self.model = self.build_model(name)
     
        self.skip = 4
        self.i = 0
        self.prev_action = 1
        self.recent_frames = []
 
    def build_model(self, name):
        # input: state
        # output: each action's Q-value
        input_shape = [self.para.img_shape[0], self.para.img_shape[1], self.para.k]
        screen_stack = tf.keras.Input(shape=input_shape, dtype=tf.float32)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4)(screen_stack) # (4, 8, 8, 32)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2)(x) # (32, 4, 4, 64)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1)(x) # (64, 3, 3, 64)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=256)(x)
        x = tf.keras.layers.ReLU()(x)
        Q = tf.keras.layers.Dense(self.para.action_num)(x)

        model = tf.keras.Model(name=name, inputs=screen_stack, outputs=Q)

        return model
 
    def max_Q(self, state):
        output = self.model(state)
        return tf.reduce_max(output, axis=1)
 
    def max_action(self, state):
        output = self.model(state)
        # return tf.reduce_max(output, axis=1)
        return tf.argmax(output, axis=1)

    def select_action(self, state):  

        # state = np.expand_dims(state, axis = 0)
        output = self.model(state)
        action = tf.argmax(output, axis=1)[0]
        action = int(action.numpy())

        return action

    def save_checkpoint(self, path):  
        print(f'- saved ckpt {path}') 
        self.model.save_weights(path)
         
    def load_checkpoint(self, path): 
        # need call once to enable load weights.
        print(f'- loaded ckpt {path}') 
        self.model(tf.random.uniform(shape=[1, self.para.img_shape[0], self.para.img_shape[1], 
                                                        self.para.k]))
        self.model.load_weights(path)



    def act(self, obs):

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
            # print(f'c state.shape: {state.shape}')
            return state

        if(self.i >= self.skip):

            self.i = 0

            if(len(self.recent_frames) >= 4): self.recent_frames.pop(0)
            self.recent_frames.append(preprocess_screen(obs))
                        

            if  np.random.rand() < 0.1:
                action = np.random.choice(para.action_num)
            else:
                # state = np.concatenate([preprocess_screen(obs)] * 4, axis=-1)
                state = stack_frames(self.recent_frames) 
                state = np.expand_dims(state, axis = 0)  
                assert state.shape == (1, para.frame_shape[0], para.frame_shape[1], para.k)            
                action = self.select_action(state)

            self.prev_action = action

            return action

        else:
            self.i += 1
            return self.prev_action







class Replay_buffer():

    def __init__(self, size=100000):
        
        self.size = size

        self.n = 0
        self.wptr = 0 
        
        self.obs = np.zeros((size, *para.frame_shape), dtype=np.uint8)
        self.action = np.zeros((size))
        self.reward = np.zeros((size))
        self.done = np.zeros((size))


    def _preprocess_frame(self, screen):

        

        def rgb2gray(rgb):  
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
 
        screen = screen[::2,::2,:]
        screen = rgb2gray(screen) 
        
        screen = screen[..., np.newaxis] # shape is (h, w, 1)
        return screen 
        

    def add_frame(self, frame):

        i = self.wptr
        self.wptr  = (self.wptr + 1) % self.size

        self.obs[i] = self._preprocess_frame(frame)

        self.n = min(self.n + 1, self.size)

        return i


    def stack_frame(self, idx):
        
         
        if(idx < para.k-1): 
            d = para.k - (idx+1) 
            out = np.concatenate([i[np.newaxis,...] for i in self.obs[-d:]] + [i[np.newaxis,...] for i in self.obs[:idx+1]], axis=3)
        else:
            _start = idx-(para.k-1)
            end = idx+1 # non-inclusive
            start = _start
            for i in range(_start, end):
                if self.done[i] > 0.5: start = i
        
            d = para.k - (end - start)
            # if(d>0):
            #     print(f'_start: {_start}, end: {end}, start: {start}, done: {self.done[_start:end]}')
            # else:
            #     print('no done')

            # out = np.concatenate([i[np.newaxis,...] for i in self.obs[start:start+1]]*d  + [i[np.newaxis,...] for i in self.obs[start:end]], axis=3)

            out = np.concatenate([np.zeros_like(self.obs[start])[np.newaxis,...]]*d  + [i[np.newaxis,...] for i in self.obs[start:end]], axis=3)
 
            # out = np.squeeze(np.stack(self.obs[idx-(para.k-1):idx+1], axis=2), axis=3)[np.newaxis,...]

        assert out.shape == (1, para.frame_shape[0], para.frame_shape[1], para.k)
        return out

       
        


    def add_effects(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = int(done)
        # if(done):
        #     print(f'self.done[idx]: {self.done[idx]}')   


    def sample(self, size):
     
        assert self.n >= size

        # if size > self.n:
        #     idxes = np.random.choice(np.arange(para.k-1, self.n), size=size)
        # else:
        idxes = np.random.choice(np.arange(para.k-1, self.n-1), size=size, replace=False)
         
        state = np.concatenate([self.stack_frame(idx) for idx in idxes], axis=0)
        action = np.array([self.action[idx] for idx in idxes])
        reward = np.array([self.reward[idx] for idx in idxes])
        state_next = np.concatenate([self.stack_frame(idx+1) for idx in idxes], axis=0)
        done = np.array([self.done[idx] for idx in idxes])

        return state, action, reward, state_next, done
 


        
 



class Trainer():

    def __init__(self, para, buf):

        self.para = para
        self.buf = buf 
        
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.para.lr)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.para.lr, rho=0.95, epsilon=0.01)
        
        self.online_agent = Agent('online', para) 
        if('ckpt_load_path' in self.para): 
            self.online_agent.load_checkpoint(self.para.ckpt_load_path)
         

        self.target_agent = Agent('target', para) 
        self.target_agent.model.set_weights(self.online_agent.model.get_weights())
 
 

    def train(self): 

        obs = env_wrapper.reset()
 
        with open("log.txt", "w") as f: f.write("")
        with open("cum_rewards.txt", "w") as f: f.write("")
        with open("eval.txt", "w") as f: f.write("")
                

        log = {'t': None, 'eps': None, 'loss': None, 'buf_n': None, 'cum_reward': None}

        cum_reward = 0

        for t in range(1, self.para.step_num+1):  

            frame_idx = self.buf.add_frame(obs) 

            eps_cur = para.eps_cur = para.eps_begin + (t / para.step_num) * (para.eps_end - para.eps_begin)
            

            if t < para.replay_start_size or np.random.rand() < eps_cur:
                action = np.random.choice(self.para.action_num)
            else:
                state = self.buf.stack_frame(frame_idx) # (1, h, w, k)
                action = self.online_agent.select_action(state)
                 
                                
            
            obs_next, reward, done, info = env_wrapper.step(action) 
            cum_reward += reward
            
            reward = min(max(reward, -1), 1)
            self.buf.add_effects(frame_idx, action, reward, done)
            
            

            if done:
                obs_next = env_wrapper.reset()            
                with open("cum_rewards.txt", "a") as f: f.write(str({'t': t, 'cum_reward': cum_reward})+'\n')
                cum_reward = 0
                
            obs = obs_next 

            if t > para.replay_start_size and t % para.learning_period == 0:

                batch = self.buf.sample(self.para.batch_size)  
                loss = self.train_step(batch)

               
                if t % (para.target_update_period * para.learning_period) == 0:
                    self.target_agent.model.set_weights(self.online_agent.model.get_weights())
 
                if t % (para.save_period * para.learning_period) == 0:
                    self.online_agent.save_checkpoint(self.para.ckpt_save_path)


                
                if t % (para.eval_period * para.learning_period) == 0:
                    # print(f'eval cum reward: {self.evaluate()}')
                    r = self.evaluate()
                    with open("eval.txt", "a") as f: f.write(str({'t': t, 'cum_reward': r})+'\n')
                    self.online_agent.save_checkpoint(f"111022533_hw2/ckpt/eval_{t}.h5")

                if t % (para.log_period * para.learning_period) == 0:
                    
                    log['t'] = t
                    log['eps'] = eps_cur
                    log['loss'] = loss
                    log['buf_n'] = self.buf.n                

                    with open("log.txt", "a") as f: f.write(str(log)+'\n')

                    log = {'t': None, 'eps': None, 'loss': None, 'buf_n': None, 'cum_reward': None}


                


    def train_step(self, batch):
 
        states, actions, rewards, states_next, terminal = batch      
         
        states = tf.convert_to_tensor(states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.int32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        states_next = tf.convert_to_tensor(states_next, tf.float32)
   
        notTerminals = tf.convert_to_tensor(1 - terminal, tf.float32)



        # max_action = self.online_agent.max_action(states_next)
        # max_action = tf.cast(max_action, tf.int32)
        # Q = self.target_agent.model(states_next)
        # # print(f'max_action.shape: {max_action.shape}, max_action: {max_action}')

        # index = tf.stack([tf.range(tf.shape(max_action)[0]), max_action], axis=1)
        # print(f'index.shape: {index}')
        
        # tar_Q = tf.gather_nd(Q, index)   
        # print(f'tar_Q.shape: {tar_Q}')     

        # tar_Q0 = self.target_agent.max_Q(states_next)
        # print(f'tar_Q0.shape: {tar_Q0}')  

        
        
        loss = self._train_step(states, actions, rewards, states_next, notTerminals)
        return loss.numpy()


    @tf.function
    def _train_step(self, state, action, reward, next_state, notTerminal):
        
        # tar_Q = self.target_agent.max_Q(next_state)

 

        max_action = self.online_agent.max_action(next_state)
        max_action = tf.cast(max_action, tf.int32)
        Q = self.target_agent.model(next_state) 
        index = tf.stack([tf.range(tf.shape(max_action)[0]), max_action], axis=1) 
        tar_Q = tf.gather_nd(Q, index)   
      
    
 
        with tf.GradientTape() as tape:
            output = self.online_agent.model(state)
            index = tf.stack([tf.range(tf.shape(action)[0]), action], axis=1)
            Q = tf.gather_nd(output, index)
          
            tar_Q *= notTerminal
            loss = tf.reduce_mean(tf.square(reward + self.para.discount_factor * tar_Q - Q))
            
            
        gradients = tape.gradient(loss, self.online_agent.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_agent.model.trainable_variables))
    
        return loss
 

    def evaluate(self): 

        print('evaluating...')

        _env = gym_super_mario_bros.make('SuperMarioBros-v0')
        _env = JoypadSpace(_env, COMPLEX_MOVEMENT)
        
        
        cum_reward = 0 
      
        n = 5
        time_limit = 120
        for i in range(n):

            obs = _env.reset()
            start_time = time.time()
                        
            while True: 

                action = self.online_agent.act(obs)
                obs, reward, done, _ = _env.step(action)                
                cum_reward += reward 


                if time.time() - start_time > time_limit:
                    print(f"Time limit reached for episode {episode}")
                    break

                if done:
                    break
                    

        _env.close()  

        return cum_reward/n

            
buffer = Replay_buffer(para.buf_size)
trainer = Trainer(para, buffer)
trainer.train()




