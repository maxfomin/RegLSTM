import numpy as np
import math
import yaml
from random import randint


def generate_data():
    with open('definitions.yml', 'r') as f:
        conf = yaml.load(f)
    data = np.zeros((conf['data_size'], conf['sequence_length'], conf['number_features']), dtype = np.float16)
    labels = np.zeros((conf['data_size'], conf['number_features']), dtype = np.float16)
    function_pool = [math.sin, math.cos]
    function_number = [1, 2]

    for ind_feature in range(conf['number_features']):
        gen_number = randint(1, len(function_number))
        gen_function = [randint(0, len(function_pool) - 1) for _ in range(gen_number + 1)]
        gen_amp = [randint(1, 10) for _ in range(gen_number + 1)]
        gen_freq = [randint(1, 100) for _ in range(gen_number + 1)]
        for ind_data in range(conf['data_size']):
            start_time = randint(1, 5000)
            time_series = [start_time + i * conf['sequence_time_diff'] for i in range(conf['sequence_length'])]
            labels[ind_data, ind_feature] = sum([gen_amp[ind] * function_pool[ind]
            (gen_freq[ind] * (time_series[-1] + conf['label_time_diff'])) for ind in gen_function])

            data_vals = []
            for ind_seq in range(conf['sequence_length']):
                data[ind_data, ind_seq, ind_feature] = sum([gen_amp[ind] * function_pool[ind]
                (gen_freq[ind] * time_series[ind_seq]) for ind in gen_function])
    return data, labels