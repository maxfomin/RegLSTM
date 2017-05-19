import data_generator
import database

def main():
    data, labels = data_generator.generate_data()
    data_struct = database.DataStruct(data, labels)
    pass


if __name__ == '__main__':
    main()
