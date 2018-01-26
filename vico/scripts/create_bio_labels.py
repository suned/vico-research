import nltk
import pandas
import ast

data = pandas.read_csv(
    'data/labels.csv',
    converters={'tokens': lambda v: ast.literal_eval(v) if v else None}
)

bio_labels = []
for _, row in data.iterrows():
    if type(row.brand) != str:
        bio_labels.append(None)
        continue
    sample_bio_labels = []
    brand = row.brand.lower().strip()
    brand_tokens = nltk.word_tokenize(brand)
    label_length = len(brand_tokens)
    i = 0
    tokens_iter = iter(row.tokens)
    for token in tokens_iter:
        if row.tokens[i:i + label_length] == brand_tokens:
            sample_bio_labels.append('B_brand')
            for _ in brand_tokens[1:]:
                sample_bio_labels.append('I_brand')
                next(tokens_iter)
                i += 1
        else:
            sample_bio_labels.append('O')
        i += 1
    assert len(sample_bio_labels) == len(row.tokens)
    bio_labels.append(sample_bio_labels)

data['brand_bio_labels'] = bio_labels
data.to_csv('data/labels.csv', index=False)
