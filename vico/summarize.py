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
    print('Train RMSE: ', rmse(data.train_loss))
    print('Test RMSE:', rmse(data.test_loss))


if __name__ == "__main__":
    run()
