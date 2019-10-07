class Trainer:

    def __init__(self, environment,
                 memory,
                 loss,
                 agent,
                 optimizer,
                 arguments):

        self.arguments = arguments
        self.optimizer = optimizer
        self.agent = agent
        self.loss = loss
        self.memory = memory
        self.environment = environment

    def train(self):
        pass

