from gym.envs.registration import register
from src.envs.yahtzee_env import YahzeeEnv

register(
    id='src/Yahtzee-v0',
    entry_point='src.envs:YahtzeeEnv',
    #max_episode_steps=300,
    kwargs={'num_dice':3, 'game_length':'short', 'scoring_options':'upper'}
)