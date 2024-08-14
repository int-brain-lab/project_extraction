"""
Creates sessions, pre-generates stim and ephys sessions
"""

import numpy as np
import pandas as pd
import warnings
from itertools import cycle
import argparse

from iblrig.misc import truncated_exponential, draw_contrast

# TODO: add these to a parameters file, save session table with parameters!
CONTRAST_LEVELS = [1.0, 0.25, 0.125, 0.0625, 0.0]
STIM_POSITIONS = [-35, 35]  # read from base_choice_world yaml instead
STIM_PROBABILITIES = [[0.8, 0.2], [0.2, 0.8]]
REWARD_VOLUMES = [[-3, -1], [1, 3]]
REWARD_PROBABILITIES = [[0.8, 0.2], [0.2, 0.8]]
CONTRAST_DRAW_TYPE = 'biased'  # TODO: should this be 'uniform'?

BLOCK_LENGTH_UNBIASED = 90
BLOCK_LENGTH_SCALE = 60
BLOCK_LENGTH_MIN = 20
BLOCK_LENGTH_MAX = 100
SESSION_LENGTH_MAX = 2000

TRIALS_TABLE_COLUMNS = [
    'contrast', 'position', 'reward_amount', 'stim_probability_left', 'reward_probability_left'
]


def make_neuromodcw_session(rng=None):
    """
    Generate a neuromodulatorChoiceWorld session.

    Parameters
    ----------
    rng : numpy.random._generator.Generator
        random nuber generator to use for drawing stimulus positions and reward
        volumes, determining block orders, and trial order permutation

    Returns
    -------
    trials : pandas.DataFrame
        table containing trial-wise information for a single session
    """
    n_conditions = len(STIM_POSITIONS) * len(CONTRAST_LEVELS)
    n_trials_per_condition = BLOCK_LENGTH_UNBIASED / n_conditions
    if n_trials_per_condition % 1 > 0.0:
        warnings.warn(
            "Unbiased block length not divisible by n conditions. "
            "Unbiased block will be truncated to produce a balanced design.",
            RuntimeWarning
        )
    conditions = np.column_stack([
        np.tile(CONTRAST_LEVELS, len(STIM_POSITIONS)),      # stim contrast
        np.repeat(STIM_POSITIONS, len(CONTRAST_LEVELS)),    # stim position
        np.full(n_conditions, 1),                           # reward volume
        np.full(n_conditions, 0.5),                         # left side stim probability
        np.full(n_conditions, 0.0)                          # left side large reward probability
    ])
    # will repeat conditions floor(n_trials_per_condition) times
    trials_unbiased = np.repeat(conditions, n_trials_per_condition, axis=0)
    if rng is None:
        rng = np.random.default_rng()
    rng.shuffle(trials_unbiased)  # only shuffles on the first dimension
    # create full trials array for session
    trials = np.full((SESSION_LENGTH_MAX, conditions.shape[1]), np.nan)
    trials[:BLOCK_LENGTH_UNBIASED] = trials_unbiased
    # draw constrasts randomly to fill remaining trials
    trials[BLOCK_LENGTH_UNBIASED:, 0] = [
        draw_contrast(
            CONTRAST_LEVELS,
            probability_type=CONTRAST_DRAW_TYPE
        )
        for trial in range(SESSION_LENGTH_MAX - BLOCK_LENGTH_UNBIASED)
    ]
    # create bias block cycler with randomized order
    stim_p_cycler = cycle(rng.permutation(STIM_PROBABILITIES))
    trial_count = BLOCK_LENGTH_UNBIASED
    while trial_count < SESSION_LENGTH_MAX:
        stim_p = next(stim_p_cycler)  # stim position probability for this block
        block_length = int(truncated_exponential(  # length for this block
            BLOCK_LENGTH_SCALE,
            BLOCK_LENGTH_MIN,
            BLOCK_LENGTH_MAX
        ))
        # truncate block length if too few trials remain in session
        block_length = min(block_length, SESSION_LENGTH_MAX - trial_count)
        # draw stim positions for the whole block
        block_positions = rng.choice(
            STIM_POSITIONS,
            size=block_length,
            p=stim_p)
        trials[trial_count:(trial_count + block_length), 1] = block_positions
        trials[trial_count:(trial_count + block_length), 3] = stim_p[0]  # store P(left)
        trial_count += block_length
    # jointly permute volumes and probabilities to starting block, create cyclers
    permutation_inds = rng.permutation(len(REWARD_VOLUMES))
    reward_p_cycler = cycle(np.array(REWARD_PROBABILITIES)[permutation_inds])
    reward_v_cycler = cycle(np.array(REWARD_VOLUMES)[permutation_inds])
    trial_count = BLOCK_LENGTH_UNBIASED
    while trial_count < SESSION_LENGTH_MAX:
        reward_p = next(reward_p_cycler)
        reward_v = next(reward_v_cycler)
        block_length = int(truncated_exponential(  # length for this block
            BLOCK_LENGTH_SCALE,
            BLOCK_LENGTH_MIN,
            BLOCK_LENGTH_MAX
        ))
        # truncate block length if too few trials remain in session
        block_length = min(block_length, SESSION_LENGTH_MAX - trial_count)
        # draw reward volumes for the whole block
        block_rewards = rng.choice(
            reward_v,
            size=block_length,
            p=reward_p)
        trials[trial_count:(trial_count + block_length), 2] = block_rewards
        trials[trial_count:(trial_count + block_length), 4] = reward_p[0]  # store P(left)
        trial_count += block_length
    # replace reward volume on trials with incongruent large-reward and stimulus sides with 1
    trials[trials[:, 1] * trials[:, 2] < 0, 2] = 1
    # get rid of side (sign) in reward volume column
    trials[:, 2] = np.abs(trials[:, 2])
    return pd.DataFrame(trials, columns=TRIALS_TABLE_COLUMNS)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_sessions', type=int, default=1)
    # TODO: contrast and block length draw need to take an rng for this to sense
    # parser.add_argument('-r', '--rng_seed', type=int, default=None)
    args = parser.parse_args()

    df_sessions = pd.DataFrame()
    for session_id in range(args.n_sessions):
        # get sequence of blocks/trials for a single session
        df_session = make_neuromodcw_session()
        df_session['session_id'] = session_id
        df_sessions = pd.concat([df_sessions, df_session])

    fpath = Path(__file__).parent.joinpath('neruomodcw_session_templates.pqt')
    df_sessions.to_parquet(fpath)
