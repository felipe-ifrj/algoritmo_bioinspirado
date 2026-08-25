class Cluster:

    def __init__(self, devices):
        self.devices = devices
        self.cost= None

    def calculate_cost(self, n_types=1, max_devices=1):
        if not self.devices:
            self.cost = 1.0
            return self.cost
  
        types_present = {}

        sum = 0.0
        for device in self.devices:
            sum += device.calculate_cost()
            types_present[device.type] = True

        sensor_absence_penalty = (1- (len(types_present) / n_types)) 
        self.cost = sum + (sum * sensor_absence_penalty)
        return sensor_absence_penalty

    
    def __str__(self):
        str = '\t\t'

        for i in range (len(self.devices)):
            sensor = self.devices [i]
            str+= sensor.__str__()
            if i < len(self.devices) -1:
                str+=' | '

        return str



