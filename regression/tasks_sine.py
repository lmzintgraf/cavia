import numpy as np
import torch


class RegressionTasksSinusoidal:
    """
    Same regression task as in Finn et al. 2017 (MAML)
    """

    def __init__(self):
        self.num_inputs = 1
        self.num_outputs = 1

        self.amplitude_range = [0.1, 5.0]
        self.phase_range = [0, np.pi]

        self.input_range = [-5, 5]

    def get_input_range(self, size=100):
        return torch.linspace(self.input_range[0], self.input_range[1], steps=size).unsqueeze(1)

    def sample_inputs(self, batch_size, *args, **kwargs):
        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        return inputs

    def sample_task(self):
        amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1])
        return self.get_target_function(amplitude, phase)

    @staticmethod
    def get_target_function(amplitude, phase):
        def target_function(x):
            if isinstance(x, torch.Tensor):
                return torch.sin(x - phase) * amplitude
            else:
                return np.sin(x - phase) * amplitude

        return target_function

    def sample_tasks(self, num_tasks, return_specs=False):

        amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], num_tasks)
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], num_tasks)

        target_functions = []
        for i in range(num_tasks):
            target_functions.append(self.get_target_function(amplitude[i], phase[i]))

        if return_specs:
            return target_functions, amplitude, phase
        else:
            return target_functions

    def sample_datapoints(self, batch_size):
        """
        Sample random input/output pairs (e.g. for training an orcale)
        :param batch_size:
        :return:
        """

        amplitudes = torch.Tensor(np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], batch_size))
        phases = torch.Tensor(np.random.uniform(self.phase_range[0], self.phase_range[1], batch_size))

        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        inputs = inputs.view(-1)

        outputs = torch.sin(inputs - phases) * amplitudes
        outputs = outputs.unsqueeze(1)

        return torch.stack((inputs, amplitudes, phases)).t(), outputs
