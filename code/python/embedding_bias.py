"""EmbeddingBias modules used in the paper (reference implementation)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.io import loadmat

# In the experiments used for this repo, embedding export was done on CPU.
device = torch.device('cpu')


def _find_pattern_indexes_in_batch(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Placeholder for the token-pattern finder used in the FastSpeech2 codebase.

    In the original experiments, this function searches `texts` (a padded batch
    of token IDs) for a list of short patterns and returns:
      - indexes_utt_in_batch
      - indexes_target_char_in_utt

    To keep this repository self-contained, we do not re-implement tokenization
    and pattern matching here.

    If you want to use `EmbeddingBiasCategorical`, provide your own
    implementation and replace this stub.
    """
    raise NotImplementedError(
        'Implement `_find_pattern_indexes_in_batch` in your TTS codebase if you want '
        'to use `EmbeddingBiasCategorical`.'
    )

class EmbeddingBias(object):
    """
    Bias Module to control acoustic params from embeddings analysis
    """
    def __init__(self, model_config):
        self.bias_vector_name = model_config["bias_vector"]["bias_vector_name"]
        self.layer_by_param = model_config["bias_vector"]["layer_by_param"]
        self.default_control_bias_array = model_config["bias_vector"]["value_by_param"]
        
    def layer_control_by_param(self, embeddings, control_bias_value, index_param, layer_index, indexes_utt_in_batch_to_apply_bias='all', indexes_target_char_in_utt_to_apply_bias='all', is_acoustic=True):
        embeddings_size = embeddings.size()
        embeddings_dim = embeddings.dim()
                
        load_bias_vector = loadmat(self.bias_vector_name) # vector name: bias_vector_by_layer
        bias_vector = load_bias_vector['bias_vector_by_layer'][layer_index-1][0][:, index_param].transpose()
        bias_size = len(bias_vector)
                
        if index_param == 0 and is_acoustic:
            bias_vector = bias_vector*(np.log(control_bias_value))
        else:
            bias_vector = bias_vector*control_bias_value
                    
        if embeddings_dim == 2: # frame by frame
            bias_vector = bias_vector[np.newaxis,:]
            bias_vector = torch.FloatTensor(bias_vector)
            bias_vector = bias_vector.to(device)
            embeddings = embeddings + bias_vector
        else:
            dim_bias = embeddings_size.index(bias_size)  
            dim_repeat = 1 if dim_bias==2 else 2
            lg_repeat = embeddings.size(dim_repeat)
                    
            zero_bias_vector = np.zeros([embeddings.size(0), embeddings.size(1), embeddings.size(2)])
            
            if len(indexes_utt_in_batch_to_apply_bias) == 0:
                return embeddings

            if indexes_utt_in_batch_to_apply_bias=='all':
                indexes_utt_in_batch_to_apply_bias = np.sort([*range(embeddings.size(0))]*lg_repeat)
            if indexes_target_char_in_utt_to_apply_bias=='all':
                indexes_target_char_in_utt_to_apply_bias = [*range(lg_repeat)]*embeddings.size(0)
            if dim_bias == 1:
                zero_bias_vector[indexes_utt_in_batch_to_apply_bias, :, indexes_target_char_in_utt_to_apply_bias] = bias_vector

            elif dim_bias == 2:
                zero_bias_vector[indexes_utt_in_batch_to_apply_bias, indexes_target_char_in_utt_to_apply_bias, :] = bias_vector

            zero_bias_vector = torch.FloatTensor(zero_bias_vector)
            zero_bias_vector = zero_bias_vector.to(device)
            embeddings = embeddings + zero_bias_vector

        return embeddings

    def layer_control(self, embeddings, control_bias_array, layer_index, indexes_utt_in_batch_to_apply_bias='all', indexes_target_char_in_utt_to_apply_bias='all'):
        if control_bias_array == self.default_control_bias_array:
            return embeddings

        for index_param, layer_index_by_param in enumerate(self.layer_by_param):
            if layer_index_by_param == layer_index:
                embeddings = self.layer_control_by_param(embeddings, control_bias_array[index_param], index_param, layer_index, indexes_utt_in_batch_to_apply_bias, indexes_target_char_in_utt_to_apply_bias)
        return embeddings

class EmbeddingBiasCategorical(EmbeddingBias):
    """
    Bias Module to control categorical params (silences, liaisons) from embeddings analysis
    """

    def __init__(self, model_config):
        super(EmbeddingBias, self).__init__()

        self.bias_vector_name = model_config["bias_vector"]["categorical_bias_vector_name"]
        self.layer_by_param = model_config["bias_vector"]["layer_by_param_categorical"]
        self.default_control_bias_array = model_config["bias_vector"]["value_by_param_categorical"]

    def layer_control_on_patterns(self, embeddings, categorical_control_bias_array, layer_index, texts):
        index_silences = 0
        index_liaisons = 1
        
        # Silence Bias
        if categorical_control_bias_array[index_silences] != self.default_control_bias_array[index_silences] and self.layer_by_param[index_silences] == layer_index:
            list_patterns_silences = np.array([
                (' ', 0),
                (',', 0),
                ('.', 0),
                ('?', 0),
                ('!', 0),
                (':', 0),
                (';', 0),
                ('§', 0),
                ('~', 0),
                ('[', 0),
                (']', 0),
                ('(', 0),
                (')', 0),
                ('-', 0),
                ('"', 0),
                ('¬', 0),
                ('«', 0),
                ('»', 0),
            ])
            [silences_indexes_utt_in_batch, silences_indexes_target_char_in_utt] = _find_pattern_indexes_in_batch(list_patterns_silences, texts)
            embeddings = self.layer_control_by_param(embeddings, categorical_control_bias_array[index_silences], index_silences, layer_index, silences_indexes_utt_in_batch, silences_indexes_target_char_in_utt, is_acoustic=False)

        if categorical_control_bias_array[index_liaisons] != self.default_control_bias_array[index_liaisons] and self.layer_by_param[index_liaisons] == layer_index:
            list_patterns_liaisons = np.array([
                ('er a', 1),
                ('er à', 1),
                ('er e', 1),
                ('er i', 1),
                ('er o', 1),
                ('er u', 1),
                ('er y', 1),
                ('t a', 0),
                ('t e', 0),
                ('t i', 0),
                ('t o', 0),
                ('t u', 0),
                ('t y', 0),
                ('n a', 0),
                ('n â', 0),
                ('n e', 0),
                ('n i', 0),
                ('n o', 0),
                ('n u', 0),
                ('n y', 0),
                ('es a', 1),
                ('es e', 1),
                ('es i', 1),
                ('es o', 1),
                ('es u', 1),
                ('es y', 1),
            ])
            [liaisons_indexes_utt_in_batch, liaisons_indexes_target_char_in_utt] = _find_pattern_indexes_in_batch(list_patterns_liaisons, texts)
            embeddings = self.layer_control_by_param(embeddings, categorical_control_bias_array[index_liaisons], index_liaisons, layer_index, liaisons_indexes_utt_in_batch, liaisons_indexes_target_char_in_utt, is_acoustic=False)

        return embeddings

