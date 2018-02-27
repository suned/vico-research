from vico import Config
from vico.console_arguments import ConsoleArguments


class ArgumentProvider(ConsoleArguments):
    config = Config()

    def get(self):
        return self.config
