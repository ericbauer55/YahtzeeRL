from gym import Env, logger, spaces
from gym.spaces import Dict, MultiBinary, MultiDiscrete, Discrete
from typing import Optional
from pathlib import Path
import numpy as np
from typing import Optional


class YahzeeEnv(Env):
    # ref: https://www.gymlibrary.dev/content/environment_creation/
    # ref: https://www.gymlibrary.dev/api/spaces/#multibinary
    metadata = {
        "render_modes": ["ansi"],
        "num_dice": 5,
        "game_length": ['short','full'],
        "scoring_options": ['upper','lower','both']
    }

    def __init__(self, num_dice:int=3, game_length:str='short', scoring_options:str='upper', render_mode: Optional[str]=None, ):
        # Input Validation
        error_msg = f'Scoring Option "{scoring_options}" is not valid--use: {self.metadata["scoring_options"]}'
        assert scoring_options in self.metadata["scoring_options"], error_msg
        error_msg = f'Game Length "{game_length}" is not valid--use {self.metadata["game_length"]}'
        assert game_length in self.metadata["game_length"], error_msg
        assert (num_dice>0) and (type(num_dice)==int), 'There must be a positive integer number of dice'
        assert render_mode is None or render_mode in self.metadata["render_modes"], 'Invalid rendering choice'
        self.render_mode = render_mode
        self.num_dice = num_dice
        self.scoring_options = scoring_options
        self.game_length = game_length

        # Setup the State Space
        self.observation_space = Dict({'roll_num':Discrete(3,start=1),
                                       'rolls':MultiDiscrete([6]*self.num_dice),
                                       'scored_options':self._init_scored_options()})
        
        # Setup the Action Space
        self.action_space = Dict({'keep':MultiBinary(self.num_dice), 'score_option':self._init_scored_options()})
    
    def _init_scored_options(self):
        scorable_options = self._get_scorable_options()
        if self.game_length=='short':
            scored_option_space = Discrete(len(scorable_options), start=0)
        elif self.game_length=='full':
            scored_option_space = MultiBinary(len(scorable_options))
        else:
            raise ValueError(f'Game Length "{self.game_length}" is not valid--use {self.metadata["game_length"]}')
        return scored_option_space
        
    def _get_scorable_options(self):
        # Determine the valid set of scoring options
        upper = [f'SU-{i}' for i in range(1,6+1)]
        lower = [f'SL-{opt}' for opt in['3Kind', '4Kind','SmStr','LgStr','FullHouse','Chance','Yahtzee']]
        if self.scoring_options=='upper':
            scorable_options = upper
        elif self.scoring_options=='lower':
            scorable_options = lower
        elif self.scoring_options=='both':
            scorable_options = upper+lower            
        return scorable_options
    
    def _get_obs(self):
        return {'roll_num':self._roll_num, 'rolls':self._rolls, 'scored_options':self._scored_options, 'current_score':self._score}
    
    def _get_info(self):
        '''This contains auxiliary or debugging information during a run'''
        return {'invalid_action':self._last_action_valid}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._roll_num = 1
        self._rolls = self.observation_space['rolls'].sample()
        if self.game_length=='short':
            self._scored_options = 0
        elif self.game_length=='full':
            mask = np.zeros(self.observation_space['scored_options'].shape, dtype=np.int8)
            self._scored_options = self.observation_space['scored_options'].sample(mask=mask) # should be all zeros
            assert np.sum(self._scored_options)==0 # TODO: remove this later
        
        observation = self._get_obs()
        self._last_action_valid = None
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        is_valid_action = self._is_valid_action(action)
        self._last_action_valid = is_valid_action
        if not is_valid_action:
            # Determine the reward
            reward = -100
            # Update the state of the game
            # NOTE: due to invalid action, keep state the same
        else:
            # Determine the reward
            pass


        
        if self.game_length==['short']:
            terminated = self._scored_options > 0 # if any option has been scored, the game is over
        elif self.game_length==['full']:
            terminated = all(self._scored_options) # if all the options have been scored, the game is over
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def _is_valid_action(action):
        '''
        ref: https://datascience.stackexchange.com/questions/61618/valid-actions-in-openai-gym 
        Because OpenAI Gym doesn't support dynamic action availability, we can do a couple of things:
        1. Embed list of valid actions in the self._get_info() dictionary (non-standard)
        2. Create a separate, public self.get_available_actions() method to return a list of actions
           for both the agent and the environment to reference. Then the agent only selects from these
           by masking all invalid actions with probability==0.0
        3. Allow the agent to try invalid actions but penalize them for trying it. This will require the
           environment to not progress its state, but simply signal an immensely unfavorable action.
           Risk (?): if the negative penalty is too much, does the agent "reward hack" itself into 
           avoiding invalid actions rather than promote the actions that maximize the end score?

        This method is meant to support Option 3
        '''
        return True

    
    def render(self):
        pass
        


