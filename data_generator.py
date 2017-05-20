import numpy as np
import math
from random import randint
import yaml


def generate_data(conf):
    data = np.zeros((conf['data_size'], conf['sequence_length'], conf['number_of_features']), dtype = np.float16)
    labels = np.zeros((conf['data_size'], conf['number_of_features']), dtype = np.float16)
    function_pool = [math.sin, math.cos, math.tan]
    function_number = [1, 2]

    for ind_feature in range(conf['number_of_features']):
        gen_number = randint(1, len(function_number))
        gen_function = [randint(0, len(function_pool) - 1) for i in range(gen_number + 1)]
        gen_amp = randint(1, 10)
        gen_freq = randint(1, 100)
        for ind_data in range(conf['data_size']):
            start_time = randint(1, 5000)
            time_series = [start_time + i * conf['sequence_time_diff'] for i in range(conf['sequence_length'])]
            labels[ind_data, ind_feature] = sum([gen_amp * function_pool[ind]
            (gen_freq * (time_series[-1] + conf['label_time_diff'])) for ind in gen_function])

            data_vals = []
            for ind_seq in range(conf['sequence_length']):
                data[ind_data, ind_seq, ind_feature] = sum([gen_amp * function_pool[ind]
                (gen_freq * time_series[ind_seq]) for ind in gen_function])
    return data, labels