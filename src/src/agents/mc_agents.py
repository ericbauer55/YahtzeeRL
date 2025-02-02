import numpy as np
from collections import defaultdict
import pandas as pd
from typing import Optional, List
from functools import reduce

class Policy:
    def __init__(self):
        # outer key = state, inner key = action, value = probability pi(s,a)
        self.policy = defaultdict(defaultdict(lambda x: 0.0))
    
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

            
        if valid_actions is not None:
            actions_mask = [True if a in valid_actions else False for a in actions]
        else:
            actions_mask = [True] * len(actions)

    def _union_action_space(self) -> List[str]:
        actions_by_state = [set(self.policy[state].keys()) for state in self.policy.keys()]
        all_actions = reduce(set.union, actions_by_state)
        return all_actions


class Agent:
    def __init__(self):
        self.pi = defaultdict(defaultdict(lambda x: 0.0))