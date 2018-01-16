import pandas
import numpy
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--file',
        type=str
    )
    return parser.parse_args()


def rmse(losses):
    mse = losses.mean()
    return numpy.sqrt(mse)


def run():
    args = parse_args()
    data = pandas.read_csv(args.file)
    for column_name in data:
        if 'loss' in column_name:
            print(column_name, ':')
            print('\tTrain RMSE : ', rmse(data[column_name]))
            print('\tTest RMSE  : ', rmse(data[column_name]))


if __name__ == "__main__":
    run()
