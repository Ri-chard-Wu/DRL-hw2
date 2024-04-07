from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# import gym

import numpy as np  
import math 
from random import shuffle
import tensorflow.keras.backend as K
import tensorflow as tf
from copy import deepcopy

import os 
import time
from PIL import Image
import importlib
# os.environ['CUDA_VISIBLE_DEVICES']=''


 

class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]
 

para = AttrDict({
    'action_num': len(COMPLEX_MOVEMENT), 
    # 'img_shape': (120, 128, 3), 
    # 'frame_shape': (120, 128, 1),
    'img_shape': (84, 84, 3), 
    'frame_shape': (84, 84, 1),    
    'k': 4,
    'skip': 4,
 
    
    'step_num': 10000000,
    'discount_factor': 0.99,
    
    # 'eps_begin': 0.2,
    # 'eps_end': 0.19,
    'eps_begin': 1.0,
    'eps_end': 0.01, 
    'eps_sched_period': 300000,

    'buf_size': 600000, 

    'batch_size': 128,
    # 'lr': 2.5e-4,
    'lr': 1e-4,
       

    'replay_start_size': 2000,
    # 'replay_start_size': 100,

    'learning_period': 4,
    'target_update_period': 100,
    'save_period': 10000, 
    'log_period': 250,
    'eval_period': 5000,
    # 'eval_period': 250,
    
    'save_video_period': 1000,
    # 'save_video_period': 20,


    'ckpt_save_path': "111022533_hw2/ckpt/checkpoint1.h5",
    'ckpt_load_path': "111022533_hw2/ckpt/checkpoint0.h5"
})



 

 

class FrameSkipEnv:
    def __init__(self, env):   
        self.env = env 
        self.skip = para.skip
        # self.episode = 0
        # self.t = 0

    def step(self, action):
        """
        - 4 steps at a time. 
        - take last obs, done and info, sum all 4 rewards.
        - clip reward between -1, 1.
        - return if encounter done before 4 steps.
        """        
        # self.t += 1

        cum_reward = 0

        for i in range(self.skip):
        
            obs_next, reward, done, info = self.env.step(action)  
            cum_reward += reward

            if done: break

        # cum_reward = min(max(cum_reward, -1), 1)
        return obs_next, cum_reward, done, info
 
    def reset(self):
        # self.t = 0
        # self.episode += 1
        return self.env.reset()




class EpisodicLifeEnv():
    def __init__(self, env):

        super().__init__() 
        
        self.env = env
        self.prev_lives = 0
        self.was_real_done = True

        self.episode = 0
        self.t = 0

    def step(self, action):

        self.t += 1

        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = info['life'] 

        if lives < self.prev_lives: 
            done = True

        self.prev_lives = lives

        return obs, reward, done, info

    def reset(self):

        self.t = 0
        self.episode += 1

        if self.was_real_done:
            obs = self.env.reset()
            self.prev_lives = 2
        else:
            obs, _, _, info = self.env.step(0)
            self.prev_lives = info['life']
        return obs




env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
 
# env =  EnvWrapper(env)
env =  FrameSkipEnv(env)
env =  EpisodicLifeEnv(env)


 
def preprocess_screen(screen): 

    def rgb2gray(rgb):  
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    img = Image.fromarray(screen)
    img = img.resize((84, 84), Image.BILINEAR)
    # img.save(f"state_resize2.jpeg")
    img = np.array(img) # (84, 84, 3)
    # print(f'img.shape: {img.shape}')

    img = rgb2gray(img) 
    img = img[..., np.newaxis] # shape is (h, w, 1)
    
    return img


 





'''

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

# Define the first Sequential model
model1 = Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten()
])

# Define the second Sequential model
model2 = Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compose the models into a single Model
input_layer = layers.Input(shape=(28, 28, 1))
output_layer = model2(model1(input_layer))  # Connect the output of model1 to the input of model2
model = Model(inputs=input_layer, outputs=output_layer)

model.summary()
'''




def sample_noise(shape):
    noise = tf.random.normal(shape)
    return noise


class NoisyLayer(tf.keras.layers.Layer):

    def __init__(self, units):
 
        self.units = units
        super(NoisyLayer, self).__init__()


    def build(self, input_shape):

        self.input_dim = input_shape[-1]

        mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(self.input_dim, 0.5),
                                                      maxval=1*1/np.power(self.input_dim, 0.5))
        sigma_init = tf.constant_initializer(0.4/np.power(self.input_dim, 0.5))

        self.w_mu = self.add_weight(shape=(self.input_dim, self.units),
                                             initializer=mu_init,
                                             trainable=True,
                                             name='w_mu')
        self.w_sigma = self.add_weight(shape=(self.input_dim, self.units),
                                             initializer=sigma_init,
                                             trainable=True,
                                             name='w_sigma')


        self.b_mu = self.add_weight(shape=(self.units,),
                                             initializer=mu_init,
                                             trainable=True,
                                             name='b_mu')
        self.b_sigma = self.add_weight(shape=(self.units,),
                                           initializer=sigma_init,
                                           trainable=True,
                                           name='b_sigma')


        super(NoisyLayer, self).build(input_shape)

        # self.kernel = None
        # self.bias = None

    def call(self, inputs):

        def f(x):
            return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))

        f_p = f(sample_noise([self.input_dim, 1]))
        f_q = f(sample_noise([1, self.units]))


        w_epsilon = f_p * f_q; # (dim_in, dim_out)
        b_epsilon = tf.squeeze(f_q) # (dim_out, )

        w = self.w_mu + tf.multiply(self.w_sigma, w_epsilon)
        b = self.b_mu + tf.multiply(self.b_sigma, b_epsilon)
        return tf.matmul(inputs, w) + b




class Agent:

    def __init__(self, name, para):   
        self.para = para 
        self.model = self.build_model(name)
     
        self.skip = para.skip
        self.i = para.skip
        self.prev_action = 1
        self.recent_frames = []
 


                    
    def build_model(self, name):
        # input: state
        # output: each action's Q-value
  
        input_shape = [self.para.img_shape[0], self.para.img_shape[1], self.para.k]
        screen_stack = tf.keras.Input(shape=input_shape, dtype=tf.float32)


        # x = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=4)(screen_stack) # (4, 8, 8, 32)
        # x = tf.keras.layers.ReLU()(x)
        # x = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2)(x) # (32, 4, 4, 64)
        # x = tf.keras.layers.ReLU()(x)
        # x = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=1)(x) # (64, 3, 3, 64)
        # x = tf.keras.layers.ReLU()(x)
        

        # x = tf.keras.layers.Flatten()(x)

        # x = tf.keras.layers.Dense(units=496)(x)
        # x = tf.keras.layers.ReLU()(x)


        # adv = tf.keras.layers.Dense(self.para.action_num)(x)
        # v = tf.keras.layers.Dense(1)(x)

        # model = tf.keras.Model(name=name, inputs=screen_stack, outputs=[adv, v])

 
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=4)(screen_stack) # (4, 8, 8, 32)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2)(x) # (32, 4, 4, 64)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=1)(x) # (64, 3, 3, 64)

        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(units=512)(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(units=512)(x)
        x = tf.keras.layers.ReLU()(x)

        x1 = NoisyLayer(512)(x)
        x1 = tf.keras.layers.ReLU()(x1)
        adv = NoisyLayer(12)(x1)


        x2 = NoisyLayer(512)(x)
        x2 = tf.keras.layers.ReLU()(x2)
        v = NoisyLayer(1)(x2)

        model = tf.keras.Model(inputs=screen_stack, outputs=[adv, v])
        


        # for layer in model.layers:        
        #     if 'Conv2D' in str(type(layer)):
        #         n = layer.weights[1].shape[0]
        #         layer.bias.assign(tf.ones((n))*0.01) 
                
        # for layer in model.layers:
        #     if 'Conv2D' in str(type(layer)):
        #         print(layer.weights[1])

        
        return model
 
  

    def q(self, state):
        adv, v = self.model(state)
        q = v + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))
        return q

    def max_Q(self, state):
        q = self.q(state) 
        return tf.reduce_max(q, axis=1)
 
    def max_action(self, state):
        q  = self.q(state) 
        return tf.argmax(q, axis=1)


    def select_action(self, state):  

        # state = np.expand_dims(state, axis = 0)
        q = self.q(state)        

        action = tf.argmax(q, axis=1)[0]
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
 

        if(self.i >= self.skip):

            self.i = 1

            if(len(self.recent_frames) >= para.k): self.recent_frames.pop(0)
            self.recent_frames.append(preprocess_screen(obs))
 
            if  np.random.rand() < 0.001:
                action = np.random.choice(para.action_num)
            else:
                d = len(self.recent_frames)
                state = np.concatenate([np.zeros_like(self.recent_frames[0])[np.newaxis,...]]*(para.k-d)  + [i[np.newaxis,...] for i in self.recent_frames], axis=3)
                assert state.shape == (1, para.frame_shape[0], para.frame_shape[1], para.k)            
                action = self.select_action(state / 255.0)

            # d = len(self.recent_frames)
            # state = np.concatenate([np.zeros_like(self.recent_frames[0])[np.newaxis,...]]*(para.k-d)  + [i[np.newaxis,...] for i in self.recent_frames], axis=3)
            # assert state.shape == (1, para.frame_shape[0], para.frame_shape[1], para.k)            
            # action = self.select_action(state / 255.0)

            self.prev_action = action

            return action

        else:
            self.i += 1
            return self.prev_action






class Replay_buffer():

    def __init__(self, trainer, size=100000):
        
        self.size = size

        self.trainer = trainer

        self.n = 0
        self.wptr = 0 
        
        self.obs = np.zeros((size, *para.frame_shape), dtype=np.uint8)
        self.action = np.zeros((size))
        self.reward = np.zeros((size))
        self.done = np.zeros((size))


    def add_frame(self, frame):

        i = self.wptr
        self.wptr  = (self.wptr + 1) % self.size

        self.obs[i] = preprocess_screen(frame)

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
            for i in range(_start, end-1):
                if self.done[i] > 0.5: start = i+1
        
            d = para.k - (end - start)

            out = np.concatenate([np.zeros_like(self.obs[start])[np.newaxis,...]]*d  + [i[np.newaxis,...] for i in self.obs[start:end]], axis=3) / 255.0
 
            # out = np.squeeze(np.stack(self.obs[idx-(para.k-1):idx+1], axis=2), axis=3)[np.newaxis,...]

        assert out.shape == (1, para.frame_shape[0], para.frame_shape[1], para.k)
        return out

        
    def add_effects(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = int(done)
     
  

    def sample(self, size):     

        assert self.n >= size
 
        idxes = np.random.choice(np.arange(para.k-1, self.n-1), size=size, replace=False)   
 
        return self.retrive_data(idxes)



    def retrive_data(self, idxes):
        # print(f'idxes: {idxes}')
        state = np.concatenate([self.stack_frame(idx) for idx in idxes], axis=0) 
        action = np.array([self.action[idx] for idx in idxes])
        reward = np.array([self.reward[idx] for idx in idxes])
        state_next = np.concatenate([self.stack_frame(idx+1) for idx in idxes], axis=0) 
        done = np.array([self.done[idx] for idx in idxes])
        
        # print(f'max(state): {np.max(state)}, state: {state}')

        state = tf.convert_to_tensor(state, tf.float32) 
        action = tf.convert_to_tensor(action, tf.int32)
        reward = tf.convert_to_tensor(reward, tf.float32)
        state_next = tf.convert_to_tensor(state_next, tf.float32)   
        done = tf.convert_to_tensor(done, tf.float32)

        return state, action, reward, state_next, done




class Logger():

    # def __init__(self):
    #     self.frame_dir = './img/'

    def save_frame(self, dir_name, name, frame):

        
        if(not os.path.exists(dir_name)):
            os.mkdir(dir_name)

        img = Image.fromarray(frame)
        # img = img.resize((84, 84), Image.BILINEAR)
        img.save(os.path.join(dir_name, name))    



logger = Logger()


class Trainer():

    def __init__(self, para):

        self.para = para
        self.buf = Replay_buffer(self, para.buf_size)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=para.lr)
     

        self.online_agent = Agent('online', para) 
        if('ckpt_load_path' in self.para): 
            self.online_agent.load_checkpoint(self.para.ckpt_load_path)

        self.target_agent = Agent('target', para) 
        self.target_agent.model.set_weights(self.online_agent.model.get_weights())
 
 

    def train(self): 

        obs = env.reset()
 
        # with open("log.txt", "w") as f: f.write("")
        # with open("cum_rewards.txt", "w") as f: f.write("")
        # with open("eval.txt", "w") as f: f.write("")
        
       

        log = {'t': None, 'eps': None, 'loss': None, 'buf_n': None, 'cum_reward': None}

        cum_reward = 0
        

        for t in range(1, self.para.step_num+1):  

            frame_idx = self.buf.add_frame(obs) 

            if(t < para.eps_sched_period):
                eps_cur = para.eps_begin + (t / para.eps_sched_period) * (para.eps_end - para.eps_begin)
            else:
                eps_cur = para.eps_end

            if t < para.replay_start_size or np.random.rand() < eps_cur:
                action = np.random.choice(self.para.action_num)
            else:
                state = self.buf.stack_frame(frame_idx) # (1, h, w, k)
                action = self.online_agent.select_action(state)

            # state = self.buf.stack_frame(frame_idx) # (1, h, w, k)
            # action = self.online_agent.select_action(state)
                                
            obs_next, reward, done, info = env.step(action) 
            cum_reward += reward

            if(env.episode % para.save_video_period == 0):
                logger.save_frame(f'img/episode-{env.episode}', f't-{env.t}.jpeg', obs_next)

            # reward = min(max(reward, -1), 1)
            # reward = np.sign(reward) * (np.sqrt(abs(reward) + 1) - 1) + 0.001 * reward
            self.buf.add_effects(frame_idx, action, reward, done)
            
            if done:
                obs_next = env.reset()            

            if env.was_real_done:
                with open("cum_rewards.txt", "a") as f: f.write(str({'t': t, 'cum_reward': cum_reward})+'\n')
                cum_reward = 0

            obs = obs_next 

            if t > para.replay_start_size and t % para.learning_period == 0:

              
                batch = self.buf.sample(self.para.batch_size)  
                losses = self.train_step(batch)
              
                if t % (para.target_update_period * para.learning_period) == 0:
                    self.target_agent.model.set_weights(self.online_agent.model.get_weights())
 
                if t % (para.save_period * para.learning_period) == 0:
                    self.online_agent.save_checkpoint(self.para.ckpt_save_path)


                if t % (para.eval_period * para.learning_period) == 0:
                    # print(f'eval cum reward: {self.evaluate()}')
                    r = self.evaluate(self.online_agent)
                    with open("eval.txt", "a") as f: f.write(str({'t': t, 'cum_reward': r})+'\n')
                    self.online_agent.save_checkpoint(f"111022533_hw2/ckpt/eval_{t}.h5")

                if t % (para.log_period * para.learning_period) == 0:
                    
                    log['t'] = t
                    log['eps'] = eps_cur
                    log['loss'] = np.mean(losses)
                    log['buf_n'] = self.buf.n                

                    with open("log.txt", "a") as f: f.write(str(log)+'\n')

                    log = {'t': None, 'eps': None, 'loss': None, 'buf_n': None, 'cum_reward': None}

 
    def train_step(self, batch): 
        losses = self._train_step(batch)
        return losses.numpy()

    

    
    def ddqn_tar_q(self, state_next):
        max_action = tf.cast(self.online_agent.max_action(state_next), tf.int32)        
        Q = self.target_agent.q(state_next) 
        index = tf.stack([tf.range(tf.shape(max_action)[0]), max_action], axis=1) 
        tar_Q = tf.gather_nd(Q, index)   
        return tar_Q


    @tf.function
    def _train_step(self, batch):

        state, action, reward, state_next, terminal = batch   

        tar_Q = self.ddqn_tar_q(state_next) * (1 - terminal)

        with tf.GradientTape() as tape:
            output = self.online_agent.q(state)
            index = tf.stack([tf.range(tf.shape(action)[0]), action], axis=1)
            Q = tf.gather_nd(output, index)
            losses = tf.square(reward + self.para.discount_factor * tar_Q - Q)
            loss = tf.reduce_mean(losses)            
        
        gradients = tape.gradient(loss, self.online_agent.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_agent.model.trainable_variables))
    
        return losses
 
 


    def evaluate(self, agent, n=5, verbose=False): 

        print('evaluating...')

        _env = gym_super_mario_bros.make('SuperMarioBros-v0')
        _env = JoypadSpace(_env, COMPLEX_MOVEMENT)
        
        
        cum_reward = 0 
      
        # n = 5
        time_limit = 120
        for i in range(1, n+1):

            obs = _env.reset()
            start_time = time.time()
                        
            while True: 

                action = agent.act(obs)
                obs, reward, done, _ = _env.step(action)                
                cum_reward += reward 


                if time.time() - start_time > time_limit:
                    print(f"Time limit reached  ")
                    break



                if done:
                    break
                    

            if verbose:
                print(f'i: {i}, avg reward: {cum_reward/i}')

        _env.close()  

        return cum_reward/n
 

trainer = Trainer(para)
trainer.train()



# module = importlib.import_module('111022533_hw2_test')
# Agent = getattr(module, 'Agent')
# agent = Agent()
# trainer.evaluate(agent, 20, True)




