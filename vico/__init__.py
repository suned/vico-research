import logging

from vico import args

log = logging.getLogger('vico')
log.setLevel(args.get().log_level)

console_handler = logging.StreamHandler()

formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] [%(name)15s] - %(message)s',
    '%H:%M:%S'
)
console_handler.setFormatter(formatter)

log.addHandler(console_handler)