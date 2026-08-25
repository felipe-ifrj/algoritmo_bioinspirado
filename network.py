class Network:

    def __init__(self, devices_available):
        self.clusters = []
        self.cost = None
        self.devices_available = devices_available

    def calculate_cost(self, scenario):
        if not self.clusters:
            return 1000.00

        total_devices_used = 0
        sum_cost = 0.0

        for cluster in self.clusters:
            cluster.calculate_cost(scenario.number_of_sensor_types, scenario.number_of_devices_per_type)
            sum_cost += cluster.cost 
            total_devices_used += len(cluster.devices)
        unused_sensors_penalty = (len(scenario.devices) - total_devices_used) * 2.0 
        cluster_count_penalty = (scenario.max_clusters() - len(self.clusters)) * 1.5
        self.cost = sum_cost + unused_sensors_penalty + cluster_count_penalty
        return self.cost                                
    
    def __str__(self):
        str = ''
        i=0
        str+=f'\t\nCost: {self.cost}'
        for cluster in self.clusters:
            str += f'\t\nCluster: {i} \n'
            str += cluster.__str__() 
            i+=1
        return str + '\n\n'