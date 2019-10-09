from old_shit.models import GeneralModel


class ParentReplay(GeneralModel):

    def __init__(self, device, **kwargs):
        super(ParentReplay, self).__init__(device, **kwargs)

    def sample(self, *args, **kwargs):
        pass  # todo: implement in childclasses (or here if shared functionality)

    def push(self, *args, **kwargs):
        pass  # todo: implement in childclasses (or here if shared functionality)

    # todo: add function template for all replays and then make classes that implement these methods
