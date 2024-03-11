# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 17:52:04 2024

@author: Julius de Clercq
"""



#%%             Imports
###########################################################

import pandas as pd
import numpy as np
import os
from time import time as t 

from collections import deque
from itertools import product
from gym import Env
from gym.spaces import Box 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Concatenate   #, LSTM, Dropout, GaussianNoise, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow as tf 
import keras

from statsmodels.tsa.api import VAR


#%%             Data preprocessing
###########################################################

# Change working directory to the directory of this script (only works when running the full script).
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Reading data.
df = pd.read_csv("ETF_data_RAW.csv", index_col = 0, header= [0,1])

# Only keeping day close prices and simplifying column names and index.
df = df.loc[:, df.columns.get_level_values('Attributes') == 'Close']
df.columns = df.columns.droplevel(level='Attributes')
df.columns = df.columns.str.replace('.US', '')

# Now selecting the ETFs with the largest price history.
ETFs_to_keep = ['TAN',    # INVESCO SOLAR ETF
                # 'GBF',        # Has some missing values here and there.
                'TFI',    # SPDR NUVEEN BLOOMBERG MUNICIPAL BOND ETF
                'UGA',    # UNITED STATES GASOLINE FUND
                'UNG',    # UNITED STATES NATURAL GAS FUND
                'XSD',    # SPDR S&P SEMICONDUCTOR ETF
                # 'PSI',        # Has some missing values here and there, and we already have a semiconductor ETF (XSD).
                'XLP',    # CONSUMER STAPLES SELECT SECTOR SPDR FUND
                'IHE'     # ISHARES US PHARMACEUTICALS ETF
                ]
df = df.loc[:, df.columns.get_level_values('Symbols').isin(ETFs_to_keep)]

# Use datetime index.
df.index = pd.to_datetime(df.index)

# Convert prices to percentage log returns. $r_t=100\log(P_t/P_{t-1})$
for column in df.columns :
    df[column] = 100*np.log(df[column]/df[column].shift(1))     
del column

# Now need to drop all data before 2008-04-21. This is the first full week of data for TAN.
df = df[df.index >= '2008-04-21']

# Should I drop March 2020? Might be more troublesome due to continuity than just keeping it in.
# df = df[('2020-03-01' > df.index) | (df.index  >= '2020-04-01')]


# Check for NA values.
# print(f"There are {df.isna().sum().sum()} NA values in the dataframe.")
# print(df[df.isna().any(axis=1)].index)
# print(f"NAs per ETF \n{df.isna().sum()}")


# The only NA values left are in 2008. As I am using this as burn-in period,
# I don't need to worry about fudging a week of rewards.
df = df.dropna()

# How many observations in 2008?
# print(f"Observations in 2008: {df[df.index.year < 2009].shape[0]}")
# Answer: 178
# This should be enough for initialization.


# Saving data
df.to_csv("ETF_data.csv")


#%%             Hedging environment
###########################################################


class ETFHedgingEnv(Env):
    def __init__(self, df, S_ETF, lag_window, burn_in_end_date, prediction_sample_start_date):
        super(ETFHedgingEnv, self).__init__()
        
       # Reorder dataframe such that the hedging target is the first column.
       # This is needed for conversion to tf tensor, as the column names will be lost.
       
        self.df = df[[S_ETF] + [col for col in df.columns if col != S_ETF]] 
        self.S_ETF = S_ETF
        
        # Define state and action spaces.
        self.action_space = Box(low=-2, high=2, shape=(self.df.shape[1] - 1,), dtype=np.float64)
        self.state_space = Box(low=-np.inf, high=np.inf, shape=(self.df.shape[1],), dtype=np.float64)
        
        self.action_dim = self.action_space.shape[0]

        self.lag_window = lag_window
        self.prediction_sample_start_date = prediction_sample_start_date
        self.train_weeks = df[df.index >= burn_in_end_date].resample('W').mean().index
        self.train_weeks = self.train_weeks[self.train_weeks < prediction_sample_start_date]
        
        self.prediction_weeks =  df[df.index >= self.prediction_sample_start_date].resample('W').mean().index
        
        self.current_step = 0

    def reset(self, predict = False):
        """
        Reset the environment to the initial state. If we are moving to prediction,
        the unique_weeks are updated to reflect the weeks in the prediction sample, 
        and the next state will be the first week of the prediction sample.

        Parameters
        ----------
        predict : bool, optional
            Boolean indicating whether we are moving to the prediction sample. 
            The default is False.

        Returns
        -------
        next_state : tensorflow tensor
            The last 'lag_window' observed days, including the current week.

        """
        self.unique_weeks = self.prediction_weeks if predict else self.train_weeks
            
        # Reset the step count.
        self.current_step = 0
        
        next_state = tf.convert_to_tensor(self._next_state())
        next_state = tf.reshape(next_state, (1, next_state.shape[0], next_state.shape[1]))
        return next_state

    def step(self, action):
        """
        Move one time step.

        Parameters
        ----------
        action : tensorflow tensor
            The most recent action taken by the actor.

        Returns
        -------
        next_state : tensorflow tensor
            The last 'lag_window' observed days, including the current week.
        reward : float
            The computed reward.
        done : bool
            Indicator of whether we have reached the last week in the episode.

        """
        # Execute one time step in the environment.
        self.current_step += 1
        # Check if we have reached the final week in the data.
        done = False if self.current_step < len(self.unique_weeks) - 1 else True
        # Get the next state.
        next_state = self._next_state()
        # Get reward based on the action and the new state.
        reward = self._compute_reward(action, next_state)
        
        # Now make ready for exporting the state to the tensorflow networks.
        next_state = tf.convert_to_tensor(next_state)
        next_state = tf.reshape(next_state, (1, next_state.shape[0], next_state.shape[1]))
        
        return next_state, reward, done
    
    def _next_state(self):
        """
        Reveal the next state as all data up until the new week end, looking as
        far back as specified by the lag window.

        Returns
        -------
        next-state : pandas dataframe
            The last 'lag_window' observed days, including the current week.
            Here we cannot convert it to a tensor yet as it would lose the datetime
            index, which is necessary for isolating the current week for computing 
            the reward.

        """
        return self.df.loc[:self.unique_weeks[self.current_step]][-self.lag_window:]
    
    def _compute_reward(self, action, next_state):
        """
        Compute the reward as negative variance of the hedged portfolio in the current week.
        Negative because higher variance is punished.

        Parameters
        ----------
        action : tensorflow tensor
            The action taken by the actor.
        next_state : tensorflow tensor
            The last 'lag_window' observed days, including the current week.

        Returns
        -------
        reward : float
            The computed reward.

        """
        # Isolate the current week's data.
        current_week = next_state[self.unique_weeks[self.current_step - 1]:]
        
        # vS is the vector of returns for the initial investment. vF is the vector
        # of returns for our hedging portfolio.
        vS = current_week[self.S_ETF]
        vF = (current_week.drop(columns = self.S_ETF) @ tf.reshape(action, (action.shape[1], action.shape[0]))).squeeze()
        
        reward = -np.var(vS - vF)  
        
        return reward




#%%             Test Hedging environment
###########################################################

def test_environment(hparams):
        
    # Initialize the environment
    env = ETFHedgingEnv(df, S_ETF, hparams["lag_window"], 
                        burn_in_end_date = hparams["burn_in_end"], 
                        prediction_sample_start_date = hparams["pred_start"])
    
    _ = env.reset()
    for _ in env.unique_weeks:
        action = env.action_space.sample()
        action = tf.reshape(action, (1, action.shape[0]))
        next_state, reward, done = env.step(action)
        # print("Observation:", state)
        # print("Reward:", reward)
        if done:
            print("Episode finished.")
            break
        
    _ = env.reset(predict=True)
    for _ in env.unique_weeks:
        action = env.action_space.sample()
        action = tf.reshape(action, (1, action.shape[0]))
        state, reward, done = env.step(action)
        # print("Observation:", state)
        print("Reward:", reward)
        if done:
            print("Episode finished.")
            break
    

# import timeit
# n_runs = 5
# execution_time = timeit.timeit(stmt = lambda: test_environment(hparams), number = n_runs)
# print(f"\nExecution time: {execution_time} seconds for {n_runs} runs.\nThat is {round(execution_time/n_runs,2)} seconds per episode.")


#%%             Deep Deterministic Policy Gradient (DDPG)
###########################################################
## See Silver et al. (2014) and Lillicrap et al. (2016)

class DDPG:
    def __init__(self, lag_window, action_dim, gamma = 0.9, tau = 0.05, eta_actor = 0.0001, eta_critic = 0.001, batch_size = 32, buffer_size = 1e4):
        
        """
        Parameters
        ----------
        state_dim : integer
            Number of dimensions of the state space. Either going to use 100 or 200 lags.
            Using the name lag_window outside the class.
        action_dim : integer
            Number of dimensions of the action space. Given by the number of ETFs 
            in the hedging portfolio.
        gamma : float in range [0,1], optional
            The discount factor, which tells us at what pace to "forget" past 
            punishments/rewards. The default is 0.9, but for this hedging problem 
            we may need to set it to zero.
        tau : float in range (0,1], optional
            Defines the updating speed of the target network to the current 
            network estimates. The default is 0.05.
        eta : float in range (0,1), optional
            The step size in gradient descent for training the networks. The default is 0.01.

        """
        self.action_dim = action_dim
        self.state_dim = (lag_window, self.action_dim + 1)
        
        self.gamma = gamma 
        self.tau = tau      
        self.eta_actor  = eta_actor 
        self.eta_critic = eta_critic 
        self.replay_buffer = deque(maxlen = int(buffer_size))
        self.batch_size = batch_size
        # Initialize actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        

        # Initialize target networks.
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        
        self._update_target_networks(tau = 1) # This line copies the weights from the actor and critic networks to the target networks for initialization.

        # Define optimizers.
        self.actor_optimizer = Adam(self.eta_actor)
        self.critic_optimizer = Adam(self.eta_critic)
        
        
    def _build_actor(self):
        """
        Construct actor network architecture. I tried to avoid it, but there is 
        no other way than to flatten the input. 
        See: https://www.tensorflow.org/tutorials/structured_data/time_series

        Returns
        -------
        actor : keras sequential model
            Densely connected neural network that will be used as the policy function.

        """
        actor = Sequential([
            Input(shape = self.state_dim, batch_size = self.batch_size),
            Flatten(),
            Dense(64, activation='relu', dtype='float64'),
            BatchNormalization(),
            Dense(32, activation='relu', dtype='float64'),
            BatchNormalization(),
            Dense(self.action_dim, kernel_initializer=GlorotUniform() 
                  , activation='tanh', dtype='float64') # tanh is a sigmoid function on the (-1,1) interval.
        ])
        return actor
    
    
    def _build_critic(self):
        """
        Construct critic network architecture. As in Lillicrap et al. (2016), I
        introduce the actions after the first hidden layer. This reduces the 
        effect of the vanishing gradient problem for the actions.
        
        Returns
        -------
        critic : keras sequential model
            Densely connected neural network that will be used as value function.

        """
        state_input = Input(shape=(self.state_dim), batch_size=self.batch_size)
        action_input = Input(shape=(self.action_dim,), batch_size=self.batch_size)
        
        x = Flatten()(state_input)
        x = Dense(128, activation='relu', dtype='float64')(x)
        x = BatchNormalization()(x)
        
        x = Concatenate()([x, action_input])
        
        x = Dense(64, activation='relu', dtype='float64')(x)
        x = BatchNormalization()(x)
        
        output = Dense(1, activation='linear', kernel_initializer=GlorotUniform(), dtype='float64')(x)
        
        critic = Model(inputs=[state_input, action_input], outputs=output)
        
        return critic

    
    def _update_target_networks(self, tau):
        """
        Here we update the target actor and critic networks. Updating speed is defined by tau.
        This updating should not go too quickly to avoid instability of estimation 
        which would prevent convergence of the estimates ("dog chasing its own tail" problem).

        Parameters
        ----------
        tau : float (0,1]
            Speed of soft updating of target networks.

        Returns
        -------
        None.

        """
        for t_actor, actor in zip(self.target_actor.weights, self.actor.weights):
            t_actor.assign(t_actor * (1.0 - tau) + actor * tau)
        for t_critic, critic in zip(self.target_critic.weights, self.critic.weights):
            t_critic.assign(t_critic * (1.0 - tau) + critic * tau)
            
            
    def init_exploration_noise(self, time_steps, theta, sigma):
        """
        Initialize matrix of exploration noise according to an Ornstein-Uhlenbeck 
        process. This matrix gives the exploration noise for the full episode.
        The parameters for the process used by Lillicrap et al. (2016) are
        theta = 0.15 and sigma = 0.2. They do not specify initial value X0, but 
        it has to be set to zero because otherwise the noise will not be zero-
        centered and would therefore introduce bias.
        """
        X0    = 0
        # theta = 0.15
        # sigma = 0.2
        
        # Define a Wiener process (i.e. Brownian motion) with mean 0 and unit variance.
        W = np.random.normal(loc = 0, scale = 1, size = (time_steps - 1, self.action_dim))
        
        X = np.zeros((time_steps, self.action_dim))
        X[0, :] = X0
        for step in range(time_steps - 1):
            X[step + 1, :] =  (1 - theta ) * X[step, :] + sigma * W[step, :]
        
        self.noise = X
    
    
    def act(self, state, step):
        """
        Determine action as the sum of the policy's decision and the exploration noise.
        """
        
        return self.actor(state) + self.noise[step]
        
            
    def train(self):
        
        # Randomly select a batch of experiences from the replay buffer.
        batch_index = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_index]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        states      = tf.concat(states, axis = 0)
        actions     = tf.concat(actions, axis = 0)
        next_states = tf.concat(next_states, axis = 0)
        rewards, dones = np.array(rewards), np.array(dones)
        
        with tf.GradientTape() as tape:
            # Set prediction targets for the critic network (i.e. Q-function)
            target_actions = self.target_actor(next_states, training=True)
            target_q_values = self.target_critic([next_states, target_actions], training=True)
            target_values = rewards + (1 - dones) * self.gamma * target_q_values
            
            # Compute the critic's loss function.
            predicted_values = self.critic([states, actions], training=True)
            critic_loss = tf.reduce_mean(tf.square(target_values - predicted_values))
            self.q_value_var = np.var(predicted_values)
        
        # Train the critic network.
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            # Compute the loss function of the 
            actions = self.actor(states, training=True)
            q_values = self.critic([states, actions], training=True)
            actor_loss = -tf.reduce_mean(q_values)

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        self._update_target_networks(self.tau)


#%%             DDPG Execution
###########################################################
# df_rewards, df_rewards_pred, agent = run_ddpg(env, hparams)
def run_ddpg(env, hparams):
    
    num_episodes = hparams["num_episodes"]
    lag_window = hparams["lag_window"]
    batch_size = hparams["batch_size"]
    buffer_size = hparams["buffer_size"]
    
    gamma = hparams["gamma"]
    tau = hparams["tau"]
    eta_actor = hparams["eta_actor"]
    eta_critic = hparams["eta_critic"]
    theta = hparams["theta"]
    sigma = hparams["sigma"]
    
    # Initialize Dataframes to track rewards.
    df_rewards      = pd.DataFrame(index = env.train_weeks[:-1],     columns = [i for i in range(num_episodes)])
    df_rewards_pred = pd.DataFrame(index = env.prediction_weeks[:-1], columns = [i for i in range(num_episodes)])
    df_qvalue_vars  = pd.DataFrame(index = env.train_weeks[:-1],     columns = [i for i in range(num_episodes)])
    
    # Initialize the DDPG agent.
    agent = DDPG(lag_window, env.action_dim, gamma = gamma, tau = tau, 
                 eta_actor = eta_actor, eta_critic = eta_critic, 
                 batch_size = batch_size, buffer_size = buffer_size)
    
    ################
    # Training
    for episode in range(num_episodes):
        print("Episode:", episode)
        start = t() # Timing
        state = env.reset(predict = False)
        agent.init_exploration_noise(len(env.unique_weeks), theta, sigma)
        
        episode_rewards = []
        episode_qvalue_var = []
        for i in range(len(env.unique_weeks) - 1):
        # for i in range(31):
            """
            Compute the action using the actor network.
            Actions lie on a [-1,1] interval due to tanh activation function. 
            However, I want to give algorithm the liberty to choose weights on 
            the [-2,2] interval. Hence, I simply scale the actions by 2.
            """
            # print(f"t = {i} \n")
            action = 2 * agent.act(state, i)     
            next_state, reward, done = env.step(action)
            agent.replay_buffer.append((state, action, reward, next_state, done))
            
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.train()
                episode_qvalue_var.append(agent.q_value_var)
            else:
                episode_qvalue_var.append(None)
                
            state = tf.identity(next_state)
            
            episode_rewards.append(reward)
            
            
        df_rewards[episode] = episode_rewards
        df_qvalue_vars[episode] = episode_qvalue_var
        
        print(f"  Completed in {int((t() - start)/60)} minutes {round((t() - start) % 60, 2)} seconds.")
        
        ################
        # Prediction
        start = t()
        state = env.reset(predict = True)
        
        pred_rewards = []
        for i in range(len(env.unique_weeks) - 1):
            action = 2 * agent.actor(state)  # Now taking the actions straight from the policy without noise.
            next_state, reward, done = env.step(action)
            pred_rewards.append(reward)
        df_rewards_pred[episode] = pred_rewards
        print(f"  Prediction took {int((t() - start)/60)} minutes {round((t() - start), 2)} seconds.\n")
        
    return df_rewards, df_rewards_pred, df_qvalue_vars, agent



#%%             Grid Search
###########################################################

def grid_search(df, S_ETF, hparams, grid_search_hparams):
    """
    Perform grid search for analysis of the effect of different hyperparameter 
    values on model performance.

    Parameters
    ----------
    df : pandas dataframe
        The preprocessed ETF return data.
    S_ETF : str
        String denoting the asset chosen as investment that is to be hedged.
    hparams : dict
        Dictionary containing the default values for all hyperparameters.
    grid_search_hparams : dict
        Dictionary of lists, containing all hyperparameters and candidate values 
        that are to be analyzed through grid search.

    Returns
    -------
    gs_results : TYPE
        DESCRIPTION.
    gs_mean_rewards : TYPE
        DESCRIPTION.
    gs_rewards_per_timestep : TYPE
        DESCRIPTION.

    """
    # hyps = hparams
    non_gridsearch_hparams = {key: value for key, value in hparams.items() if key not in grid_search_hparams}
    hyperparameter_combinations = [i for i in product(*grid_search_hparams.values())]    
    gshs_keys = list(grid_search_hparams.keys())
    
    print(f"\nStarting evaluation of {len(hyperparameter_combinations)} hyperparameter sets, with a total of {len(hyperparameter_combinations) * hparams['num_episodes']} episodes.")
    estimate_completion_time(hparams["num_episodes"], seconds_per_episode = 125, runs = len(hyperparameter_combinations))
    
    gs_results = {}
    
    j = 0
    for hyp_set in hyperparameter_combinations:
        print(f"\nHyperparameter set: {j}")
        j += 1
        
        # Reconstructing hyperparameter set.
        grid_search_hyp_set = dict(zip(gshs_keys, hyp_set))
        hyps = {**non_gridsearch_hparams, **grid_search_hyp_set}
        
        env = ETFHedgingEnv(df, S_ETF, hyps["lag_window"], 
                            burn_in_end_date = hyps["burn_in_end"], 
                            prediction_sample_start_date = hyps["pred_start"])
        
        df_rewards, df_rewards_pred, df_qvalue_vars, agent = run_ddpg(env, hyps)
        
        mean_rewards = np.array([df_rewards[i].mean() for i in df_rewards.columns])
        mean_rewards_pred = np.array([df_rewards_pred[i].mean() for i in df_rewards_pred.columns])
        
        results = {str(hyps):  {"agent"              : agent,
                                "df_rewards"         : df_rewards,
                                "df_rewards_pred"    : df_rewards_pred,
                                "mean_rewards"       : mean_rewards,
                                "mean_rewards_pred"  : mean_rewards_pred,
                                "df_qvalue_vars"     : df_qvalue_vars
                    }}
        
        gs_results.update(results)
    
    # Store results in convenient formats and save locally.
    gs_mean_rewards = {}
    for res in ["mean_rewards", "mean_rewards_pred"]:
        gs_mean_rewards[res] = np.array([result[res] for result in gs_results.values()]).transpose()
        gs_mean_rewards[res] = pd.DataFrame(gs_mean_rewards[res], columns = [str(hprms) for hprms in gs_results.keys()])
        gs_mean_rewards[res].index.name = "Episode"
    
    # idx = 0
    # while True:
    #     file_name = f'gs_mean_rewards_{idx}.pkl'
    #     # Check if the file already exists
    #     if not os.path.exists(file_name):
    #         # Save the dictionary to a pickle file
    #         with open(file_name, 'wb') as f:
    #             pd.to_pickle(gs_mean_rewards, f)
    #             break
    #     else:
    #         idx +=1
        
    gs_rewards_per_timestep = {}
    for res in ["df_rewards", "df_rewards_pred"]:
        gs_rewards_per_timestep[res] = {}
        for result in gs_results.items():
            gs_rewards_per_timestep[res].update({str(result[0]): result[1][res]})
    
    gs_q_value_var = {}
    for result in gs_results.items():
        gs_q_value_var.update({str(result[0]): result[1]["df_qvalue_vars"]})
    
    # idx = 0
    # while True:
    #     file_name = f'gs_rewards_per_timestep_{idx}.pkl'
    #     # Check if the file already exists
    #     if not os.path.exists(file_name):
    #         # Save the dictionary to a pickle file
    #         with open(file_name, 'wb') as f:
    #             pd.to_pickle(gs_mean_rewards, f)
    #             break
    #     else:
    #         idx +=1
            
    return gs_results, gs_mean_rewards, gs_rewards_per_timestep, gs_q_value_var



#%%             Estimate completion time
###########################################################

def estimate_completion_time(episodes, seconds_per_episode = 125, runs = 1):
    exp_completion_time = seconds_per_episode * hparams["num_episodes"] * runs
    print(f"Assuming {seconds_per_episode} seconds per episode, this is expected to take: {int(exp_completion_time/3600)} hours, {int((exp_completion_time % 3600)/60)} minutes and {exp_completion_time % 60 } seconds.")



#%%             Check critic output from the grid search results
###########################################################


def check_gs_critic(gs_index):
        
    i=0
    for item in gs_results.items():
        if i == gs_index:
            key = item[0]
        i+=1
        
    self = gs_results[key]["agent"]
    
    
    
    # Randomly select a batch of experiences from the replay buffer.
    batch_index = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
    batch = [self.replay_buffer[i] for i in batch_index]
    
    states, actions, rewards, next_states, dones = zip(*batch)
    states      = tf.concat(states, axis = 0)
    actions     = tf.concat(actions, axis = 0)
    next_states = tf.concat(next_states, axis = 0)
    rewards, dones = np.array(rewards), np.array(dones)
    
    return self.critic([states, actions], training=True)


#%%             VAR forecast rewards
###########################################################
# VAR_rewards_pred = VAR_benchmark(df, S_ETF, hparams, fit_window_size)
def VAR_benchmark(df, S_ETF, hparams, fit_window_size):
    
    

    env = ETFHedgingEnv(df, S_ETF, hparams["lag_window"], 
                        burn_in_end_date = hparams["burn_in_end"], 
                        prediction_sample_start_date = hparams["pred_start"])
    
    # Initialize Dataframes to track rewards.
    VAR_rewards_pred = pd.DataFrame(index = env.prediction_weeks[:-1])
    
    
    
    state = env.reset(predict=True)
    
    
    
    VAR_rewards = []
    for i in range(len(env.unique_weeks) - 1):
        # Removing the batch dimension of the tensor and converting to np array.
        state = state[0].numpy()     
        
        
        model = VAR(state).fit()
        
        forecast = model.forecast(state[-model.k_ar:], steps=5)
        fitting_window = np.concatenate([state[-fit_window_size:], forecast])
        
        S = fitting_window[:,0]
        F = fitting_window[:,1:]
        omega = np.linalg.lstsq(F, S, rcond=None)[0]
        H = S - F @ omega
        np.var(H)
        
        action = tf.constant(omega, dtype=tf.float64)
        action = tf.reshape(action, (1, action.shape[0]))
        
        state, reward, done = env.step(action)
        
        VAR_rewards.append(reward)
        
        
    VAR_rewards_pred[0] = VAR_rewards
    
    
    # Fit linear regression on the forecast and the last 10 or so observations.
    
    # forecast will contain the forecasted values for the next 5 steps
    return VAR_rewards_pred


    
#%%             Main
###########################################################


################
# Magic numbers    

# Set randomization seed.
rseed = 98765
np.random.seed(rseed)
tf.keras.utils.set_random_seed(rseed)

# The investment we need to hedge. We use the remaining ETFs for hedging.
S_ETF = 'XLP'       # Consumer staples ETF.

# Hyperparameters
hparams =  {"gamma"       : 0,
            "tau"         : 0.01,
            "eta_actor"   : 0.0001,
            "eta_critic"  : 0.001,
            "sigma"       : 0.2,
            "theta"       : 0.15,
            "num_episodes": 5,
            "lag_window"  : 150,
            "batch_size"  : 32,
            "buffer_size" : 1e6,
            "burn_in_end" : '2009-01-01',
            "pred_start"  : '2023-01-01'
            }

VAR_fit_window_size = 25

###############
# Grid search
# Hyperparameters for grid search
grid_search_hparams = {"eta_critic"   : [0.01, 0.001, 0.0001]
                      }

gs_results, gs_mean_rewards, gs_rewards_per_timestep, gs_q_value_var = grid_search(df, S_ETF, hparams, grid_search_hparams)


VAR_rewards_pred = VAR_benchmark(df, S_ETF, hparams, VAR_fit_window_size)
VAR_rewards_pred.mean()



    


# ###############
# # Initialization
# env = ETFHedgingEnv(df, S_ETF, hparams["lag_window"], 
#                     burn_in_end_date = hparams["burn_in_end"], 
#                     prediction_sample_start_date = hparams["pred_start"])

################
# Training and Estimation
# estimate_completion_time(hparams["num_episodes"], seconds_per_episode = 125)

# df_rewards, df_rewards_pred, df_qvalue_vars, agent = run_ddpg(env, hparams)

# mean_rewards = np.array([df_rewards[i].mean() for i in df_rewards.columns])
# mean_rewards_pred = np.array([df_rewards_pred[i].mean() for i in df_rewards_pred.columns])



# What if there is a holiday, such as Christmas, on which there is no trading? 

# def save_agent(agent):
        
#     agent_data = {'critic_weights': agent.critic.get_weights(),
#                   'actor_weights': agent.actor.get_weights(),
#                   'replay_buffer': agent.replay_buffer
#                   }
    
#     file_name = 'agent_stalled_episode300.pkl'
#     # Check if the file already exists
#     with open(file_name, 'wb') as file:
#         pickle.dump(agent_data, file)

# import pickle 

# idx = 0




#%%             Plot the rewards per timestep and benchmark


# asdf= gs_rewards_per_timestep["df_rewards_pred"]
# new_names = ["0.01", "0.001", "0.0001"]
# asdf = {new_name: dataframe for new_name, (_, dataframe) in zip(new_names, asdf.items())}
# asdf.update({"VAR-OLS": VAR_rewards_pred})

# plt.figure(figsize=(12, 6))  # Adjust size as needed

# for key, dataframe in asdf.items():
#     # Calculate mean and range
#     mean = dataframe.mean(axis=1)
#     lower_bound = dataframe.min(axis=1)
#     upper_bound = dataframe.max(axis=1)

#     # Plotting mean as a line plot
#     # plt.plot(mean, label=f'${key}$')
#     if key != "VAR-OLS":
#         plt.plot(mean, label= r'$\eta_Q$' + f' = {key}')
#     else:
#         plt.plot(mean, label= f'{key}')
#     # Filling the range
#     plt.fill_between(mean.index, lower_bound, upper_bound, alpha=0.3)

# plt.ylabel('Reward')
# plt.legend()
# plt.grid(True)

# plt.savefig("GS Pred reward range"".pdf")
# plt.show()




#%%             Scrap code
###########################################################



# # Load the dictionary from the pickle file
# with open('gs_mean_rewards.pkl', 'rb') as f:
#     data_dict = pd.read_pickle(f)



# self = gs_results[asdf]["agent"]

# asd = 0
# dsa = 1
# shorten = 25
# for asd in range(len(actions) - shorten ):
#     for dsa in range(len(states) - shorten ):
#         action = actions[asd]
#         state = states[dsa]

#         pad_action = tf.pad(action, [[1, 0]])
#         pad_action  = tf.reshape(pad_action , (1, 1, pad_action.shape[0]))
#         state = tf.reshape(state, (1, state.shape[0], state.shape[1]))
#         state_action = tf.concat([state, pad_action], axis = 1)

#         q_values = self.critic(state_action, training=True)
#         print("\n")
#         print(f"Action: {action}")
#         print(f"Q-value: {q_values}")




# if np.abs(sum(action)) > sum_max:
#     action_space.remove(action)
# end = t()
# loop_time = (end-start)*len(action_space)
# print(f"Expected loop time: {round(loop_time)} seconds or {round(loop_time/60, 1)} minutes or {round(loop_time/3600, 2)} hours.")






# def _build_critic(self):
#     """
#     Construct critic network architecture. 
    
#     Returns
#     -------
#     critic : keras sequential model
#         Densely connected neural network that will be used as value function.

#     """
#     State_input = Input(shape = self.state_dim, batch_size = self.batch_size)
#     Action_input = Input(shape = self.action_dim, batch_size = self.batch_size)
#     critic = Sequential([
#         State_input,
#         Flatten(),
#         Dense(128, activation='relu', dtype='float64'),
#         BatchNormalization(),
#         Concatenate()[Action_input],
#         Dense(32, activation='relu', dtype='float64'),
#         BatchNormalization(),
#         Dense(1, dtype='float64')
#     ])
#     return critic


# def _concat_state_action(self, state, action):
#     """
#     Concatenate the current state and action for passing it to the critic 
#     network. This first requires padding the action such that it becomes a 
#     vector of equal length to the state's column dimension. Then the state 
#     and action can be concatenated.
    
#     The if statement filters for whether we are only concatenating a single 
#     state and action (need this for testing), in which case action is a 
#     vector (and len(action.shape)==1), or whether we are concatenating 
#     a batch of states and actions.

#     Parameters
#     ----------
#     state : tensorflow tensor
#         Current state.
#     action : tensorflow tensor
#         The action taken.

#     Returns
#     -------
#     state_action : tensorflow tensor
#         The concatenated state and action.

#     """
#     if len(action.shape) == 1:
#         assert len(state.shape) == 2
#         pad_action = tf.pad(action, [[1, 0]])
#         pad_action  = tf.reshape(pad_action , (1, 1, pad_action.shape[0]))
#         state = tf.reshape(state, (1, state.shape[0], state.shape[1]))
#         state_action = tf.concat([state, pad_action], axis = 1)
#     else:
#         pad_action = tf.pad(action, [[0, 0], [1, 0]])
#         pad_action  = tf.reshape(pad_action , (pad_action.shape[0], 1, pad_action.shape[1]))
#         state_action = tf.concat([state, pad_action], axis = 1)
    
#     return state_action





