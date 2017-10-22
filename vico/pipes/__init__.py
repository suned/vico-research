from pipe import Pipe, as_list

from numpy import ndarray, array


@Pipe
def as_array(values) -> ndarray:
    return array(list(values))



