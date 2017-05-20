import data_generator
import database
# import tensorflow as tf
import yaml


def main():
    with open('definitions.yml', 'r') as f:
        conf = yaml.load(f)

    data, labels = data_generator.generate_data(conf)
    data_struct = database.DataStruct(data, labels)




if __name__ == '__main__':
    main()
