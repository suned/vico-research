from .. import preprocess
from pipe import Pipe

html_tokenize = Pipe(preprocess.html_tokenize)
lowercase = Pipe(preprocess.lowercase)
remove_useless_tags = Pipe(preprocess.remove_useless_tags)
maxlen = Pipe(preprocess.maxlen)
