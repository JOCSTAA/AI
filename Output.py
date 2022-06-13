import Speak

class Output():
    def __init__(self):
        self.mouth = Speak.Speak()
        self.mouth.start_speaking()

    def act(self, sound):
        self.mouth.write(sound)