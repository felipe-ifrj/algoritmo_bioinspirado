import random


class Snapshot:
    
    def __init__(self, id, type, weigth, mean, variance, original=True):
        self.id = id
        self.type = type
        self.weight = weigth
        self.mean = mean
        self.variance = variance
        self.std_dev = variance ** 0.5
        self.original = original
        
    
    def clone_and_simulate(self, id_sim, simulation_variance):
        mean_noise = random.gauss(0, simulation_variance)
        
        if mean_noise < 0:
            mean = self.mean - (self.mean * mean_noise)
            if mean < 0:
                 mean = 0
        else:
            mean = self.mean + (self.mean * mean_noise)
        
        std_noise = random.gauss(0, simulation_variance)
        if std_noise < 0:
            std_dev = self.std_dev - (self.std_dev * std_noise)
            if std_dev < 0:
                 std_dev = 0
        else:
            std_dev = self.std_dev + (self.std_dev * std_noise)

        weight_noise = random.gauss(0, simulation_variance)
        if weight_noise < 0:
            weigth = self.weight - (self.weight * weight_noise)
            if weigth < 0:
                 weigth = 0
        else:
            weigth = self.weight + (self.weight * weight_noise)

        return Snapshot(f'{self.type}_{id_sim}',self.type, weigth, mean, std_dev ** 2, False)

