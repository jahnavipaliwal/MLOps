from metaflow import FlowSpec, step, Parameter

class Counterflow(FlowSpec):

    begin_count = Parameter('ct', default = 20, type = int, required = True)

    @step
    def start(self):
        self.count = self.begin_count
        self.next(self.add)

    ...
