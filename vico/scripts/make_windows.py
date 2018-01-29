import pickle
import ast


with open('data/indices.pkl', 'rb') as f:
    indices = pickle.load(f)


labels = labels = pandas.read_csv('data/labels.csv', converters={'tokens': lambda v: ast.literal_eval(v) if v else []})
