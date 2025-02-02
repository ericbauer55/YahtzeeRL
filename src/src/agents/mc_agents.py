import numpy as np
from collections import defaultdict
import pandas as pd
from typing import Optional, List, TypeVar
from functools import reduce
from dataclasses import dataclass, field

class Policy:
    def __init__(self, epsilon:float=0.01):
        # outer key = state, inner key = action, value = probability pi(s,a)
        self.policy = defaultdict(defaultdict(lambda x: 0.0))
        self.epsilon = epsilon # probability to sample off-policy in a uniform way
    
    def sample_action(self, state:str, valid_actions:Optional[List[str]])->str:
        '''
        This method samples the policy pi given state s
        If state s doesn't exist in self.policy (i.e. self.policy[state] is an empty defaultdict)
        Then, uniformly select from valid_actions if it is not none.
        If it there are no valid_actions elicited from the environment, then
        this sampling should occur after UNIONing the set of all actions seen across other states
        '''
        # Determine the list of actions to sample from
        actions = list(self.policy[state].keys())
        if len(actions)==0: # this state has no defined policy yet
            if valid_actions is not None: # there is a set of valid actions to sample from
                actions = valid_actions
            else: # there is no set of actions to sample from
                actions = self._union_action_space()
                for a in actions: # use the action_space so far to set a uniform action distribution for state
                    self.policy[state][a] = 1.0 / len(actions) 

        # Filter which actions will be sampled from and which ones will have 0.0 probability
        if valid_actions is not None:
            actions_mask = [True if a in valid_actions else False for a in actions]
        else:
            actions_mask = [True] * len(actions)
        
        # Determine the action sampling probabilities
        # Check epsilon exploration (which will provide uniform action sampling)
        if np.random.choice(['off-policy','greedy on-policy'], p=[self.epsilon, 1-self.epsilon])=='off-policy':
            action_prob = [1.0/sum(actions_mask) if masked else 0.0 for masked in actions_mask]
        else:
            action_prob = [self.policy[state][a] if masked else 0.0 for a, masked in zip(actions, actions_mask)]
            # normalize the action probabilities so sum(valid_actions_probs) = 1.0
            action_prob = [p/sum(action_prob) for p in action_prob]
        
        # Sample the action
        sampled_action = np.random.choice(actions, p=action_prob)
        return sampled_action


    def _union_action_space(self) -> List[str]:
        actions_by_state = [set(self.policy[state].keys()) for state in self.policy.keys()]
        all_actions = reduce(set.union, actions_by_state)
        return all_actions

    def renormalize_action_space(self) -> None:
        '''
        This method guarantees that within each state of the policy, the action probabilities sum to 1.0
        This is relevant as different actions are added to the self.policy[state] dictionary
        '''
        for state in self.policy.keys():
            action_probs = [p for p in self.policy[state].values()]
            for action in self.policy[state].keys():
                self.policy[state][action] = self.policy[state][action] / sum(action_probs)

S = TypeVar('S', str, dict, int)
A = TypeVar('A', str, int)
R = TypeVar('R', int, float)

@dataclass
class Timestep:
    state: S
    action: A
    reward: R

class Trajectory:
    def __init__(self):
        self.trajectory: List[Timestep] = []
    
    def append(self, timestep_sar: Timestep):
        self.trajectory.append(timestep_sar)


class Agent:
    def __init__(self, env, epsilon=0.01):
        self.policy = Policy(epsilon=epsilon)
        self.env = env
        self.quality = defaultdict(defaultdict(lambda x:0.0))
        self.episode_log: List[Trajectory] = []

    def play_episodes(self, m_episodes:int=100):
        pass