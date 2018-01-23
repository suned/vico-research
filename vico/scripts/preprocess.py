from vico import Config
from .. import read, preprocess, configure_root_logger
from ..validate import limit
import logging
import pandas

config = Config(log_level=logging.INFO)
configure_root_logger(config)
tokenizations = (read.all_docs() >> preprocess.pipeline)(config)
data = pandas.read_csv('data/labels.csv')
data['tokens'] = None
data = data.set_index('path')
for tokenization in tokenizations:
    try:
        data.at[tokenization.document.path, 'tokens'] = tokenization.tokens
    except:
        import ipdb
        ipdb.sset_trace()
data = data.reset_index()
data.to_csv('data/labels.csv', index=False)
