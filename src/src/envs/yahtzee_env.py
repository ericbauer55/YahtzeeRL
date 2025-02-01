from gym import Env, logger, spaces
from gym.spaces import Dict, MultiBinary, MultiDiscrete, Discrete
from typing import Optional, List
from pathlib import Path
import numpy as np
from typing import Optional
import re
from collections import Counter
from pprint import pprint
from copy import deepcopy


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
                                       'rolls':MultiDiscrete([6]*self.num_dice, dtype=np.int8),
                                       'scored_options':self._init_scored_options()})
        
        # Setup the Action Space
        self.action_space = Dict({'keep':MultiBinary(self.num_dice), 'score_option':self._init_scored_options()})
        self.action_pattern = r'^([kK][01]{'+ f'{self.num_dice}'+'}|'+ '|'.join(self._get_scorable_options().keys()) +')$'
        self.action_format_checker = re.compile(self.action_pattern)
    
    def _init_scored_options(self):
        scorable_options: dict = self._get_scorable_options()
        if self.game_length=='short':
            scored_option_space = Discrete(len(scorable_options.keys()), start=0)
        elif self.game_length=='full':
            scored_option_space = MultiBinary(len(scorable_options.keys()))
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
        return {option:i for i, option in enumerate(scorable_options)}
    
    def _get_obs(self):
        return {'roll_num':self._roll_num, 'rolls':self._rolls, 'scored_options':self._scored_options}
    
    def _get_info(self):
        '''This contains auxiliary or debugging information during a run'''
        return {'invalid_action':self._last_action_valid, 'score':self._episode_score}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._roll_num = 1
        self._rolls:np.ndarray = self.observation_space['rolls'].sample().astype(np.int8)
        if self.game_length=='short':
            self._scored_options = 0
        elif self.game_length=='full':
            mask = np.zeros(self.observation_space['scored_options'].shape, dtype=np.int8)
            self._scored_options = self.observation_space['scored_options'].sample(mask=mask) # should be all zeros
            assert np.sum(self._scored_options)==0 # TODO: remove this later
        
        observation = self._get_obs()

        self._last_action_valid = None
        self._episode_score = 0
        info = self._get_info()

        self.render()

        return observation, info
    
    def step(self, action):
        is_keep_action = self._is_keep_action(action)
        is_valid_action = self._is_valid_action(action)
        self._last_action_valid = is_valid_action
        if not is_valid_action:
            # Determine the reward
            reward = -100
            # Update the state of the game
            # NOTE: due to invalid action, keep state the same
        else:
            # Determine the reward
            if is_keep_action:
                reward = 0
            else:
                reward = self._get_rolls_score(action)
            
            # Update the state of the game
            if is_keep_action:
                keep_per_die = [int(x) for x in action.replace('K','')]
                self._roll_dice(keep_per_die=keep_per_die)
            else:
                self._record_score_usage(action)
                self._roll_num = 0 # this will be --> 1 when rolling dice
                self._roll_dice(keep_per_die=None)
        
        self._episode_score += reward # for self.info
        
        if self.game_length=='short':
            terminated = self._scored_options > 0 # if any option has been scored, the game is over
        elif self.game_length=='full':
            terminated = all(self._scored_options) # if all the options have been scored, the game is over
        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return observation, reward, terminated, False, info
    
    def _get_rolls_score(self, action:str):
        assert action in self._get_scorable_options().keys(), f'Action "{action}" is not a valid scoring option'
        rolls = sorted(list(self._rolls+1))
        roll_str = ''.join([str(x) for x in rolls])
        roll_values = Counter(rolls)
        suffix = action.split('-')[-1]

        is_small_straight = (roll_str=='1234') or (roll_str=='2345') or (roll_str=='3456')
        is_large_straight = (roll_str=='12345') or (roll_str=='23456')
        if len(roll_values.most_common())>1:
            is_full_house = roll_values.most_common()[0][1]==3 and roll_values.most_common()[1][1]==2
        else:
            is_full_house = False

        if action.startswith('SU'):
            score = sum([x for x in rolls if x==int(suffix)])
        elif action.startswith('SL'):
            # Note: some of these are impossible to score with <5 dice, and will result in 0 score
            # This means those are neutral dumping grounds for bad rolls
            score = 0
            if suffix=='3Kind' and roll_values.most_common()[0][1]>=3:
                score = sum(rolls)
            elif suffix=='4Kind' and roll_values.most_common()[0][1]>=4:
                score = sum(rolls)
            elif suffix=='SmStr' and is_small_straight:
                score = 30
            elif suffix=='LgStr' and is_large_straight:
                score = 40
            elif suffix=='FullHouse' and is_full_house:
                score = 25
            elif suffix=='Chance':
                score = sum(rolls)
            elif suffix=='Yahtzee' and roll_values.most_common()[0][1]>=5:
                score = 50
        return score

    
    def _record_score_usage(self, action):
        assert action in self._get_scorable_options().keys(), f'Action "{action}" is not a valid scoring option'
        scored_action_index = self._get_scorable_options()[action]
        if self.game_length=='short':
            self._scored_options = scored_action_index # Discrete space
        elif self.game_length=='full':
            self._scored_options[scored_action_index] = 1 # MultiBinary space

    
    def _roll_dice(self, keep_per_die:Optional[List[int]]=None):
        if keep_per_die is None:
            keep_per_die = [0] * self.num_dice

        N_SIDES_PER_DIE = 6
        current_rolls:np.ndarray = self._rolls
        reroll_mask = np.ones((N_SIDES_PER_DIE,), dtype=current_rolls.dtype) # this allows all possible values to occur (re-roll)
        blank_keep_mask = np.zeros((N_SIDES_PER_DIE,), dtype=current_rolls.dtype) # this will be 1-hot encoded to keep a value
        masks = []
        for i,keep in enumerate(keep_per_die):
            if keep==True:
                mask = blank_keep_mask.copy()
                kept_event = current_rolls[i]
                mask[kept_event] = 1
            else:
                mask = reroll_mask.copy()
            masks.append(mask)
        self._roll_num += 1
        self._rolls = self.observation_space['rolls'].sample(mask=tuple(masks))
    
    def _is_valid_action(self, action:str):
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
        if self.game_length=='short':
            unused_score_options = list(self._get_scorable_options().keys())
        elif self.game_length=='full':
            unused_score_options = [option for option,state_index in self._get_scorable_options().items() 
                                    if self._scored_options[state_index]==0]
        
        if (self._is_keep_action(action)) and (self._roll_num < 3):
            return True
        elif action in unused_score_options:
            return True
        else:
            return False
    
    def _is_keep_action(self, action:str):
        pattern_keep = r'[kK][01]{' + str(self.num_dice) + '}'
        if (re.match(pattern_keep, action) is not None):
            return True
        else:
            return False
    
    def render(self):
        if self.render_mode=='ansi':
            self._render_text()
    
    def _render_text(self):
        observation = deepcopy(self._get_obs())
        observation['rolls'] = observation['rolls']+1 # index starts at 0, so move range into 1-6
        info = self._get_info()
        print('\n'+'='*80)
        print('Observation:')
        pprint(observation)
        print('-'*70)
        print('Info:')
        pprint(info)
        


