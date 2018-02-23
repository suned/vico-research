from vico import Config
from vico.console_arguments import ConsoleArguments


class CommandLineConfig(ConsoleArguments):
    config = Config()

    def get(self):
        return self.config
