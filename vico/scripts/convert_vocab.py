import pickle

with open('data/transformed_embedding.pkl', 'rb') as f:
    vocab = pickle.load(f)

n_tokens = len(vocab)
d = 300

newline = '\n'
lines = [str(n_tokens) + ' ' + str(d) + newline]
for word, vector in vocab.items():
    line = word + ' ' + ' '.join([str(v) for v in vector]) + newline
    lines.append(line)

with open('data/transformed_embedding.word2vec', 'w') as f:
    f.writelines(lines)
