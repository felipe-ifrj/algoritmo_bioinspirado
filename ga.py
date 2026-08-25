import logging
import random
from cluster import Cluster
from network import Network
import copy


logging.basicConfig(level=logging.INFO)

class GA:
    
    def __init__(self, n_population = 10, scenario = None, mutation_rate = 0.4, crossover_rate = 0.7, n_gen = 100, max_cluster_size = 0):
        self.n_population = n_population
        self.population = []
        self.scenario = scenario
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_gen = n_gen
        self.max_cluster_size = max_cluster_size
        self.generate_initial_population()
        self.best_solution()

    def generate_initial_population(self):
        for i in range(self.n_population):
            network = Network(copy.deepcopy(self.scenario.devices))
            devices_available = network.devices_available
            clusters_created = 0
            total_clusters = random.randint(1, self.scenario.max_clusters())
            while clusters_created < total_clusters:
                cluster, du, devices_available = self.generate_random_cluster(devices_available)
                if cluster is None:
                    break
                clusters_created += 1
                network.clusters.append(cluster)
            network.devices_available = devices_available
            self.population.append(network)

    def generate_random_cluster(self, devices_available):
        devices = []

        if len(devices_available) == 0:
            return None, 0, devices_available

        if len(devices_available) == 1:
            n_sensors = 1
        else:
            n_sensors = random.randint(1, len(devices_available))

        for _ in range(n_sensors):
            if len(devices_available) == 0:
                break
            index = random.randint(0, len(devices_available) - 1)
            devices.append(copy.deepcopy(devices_available[index]))
            del devices_available[index]
        cluster = Cluster(devices)
        return cluster, len(devices), devices_available

    
    def crossover(self, chromosome1, chromosome2):
        min_len = min(len(chromosome1.clusters), len(chromosome2.clusters))
        if min_len < 2:
            return copy.deepcopy(chromosome1), copy.deepcopy(chromosome2)
        
        cut = random.randint(1, min_len - 1)

        def create_child(p1, p2, cut_point):
            child = Network(None)
            child.clusters = copy.deepcopy(p1.clusters[:cut_point] + p2.clusters[cut_point:])
            
            used_ids = set()
            clean_clusters = []
            
            for cluster in child.clusters:
                new_devices = []
                for d in cluster.devices:
                    if d.id not in used_ids:
                        new_devices.append(d)
                        used_ids.add(d.id)
                
                if new_devices:
                    cluster.devices = new_devices
                    clean_clusters.append(cluster)
            
            child.clusters = clean_clusters
            
            all_ids = {d.id for d in self.scenario.devices}
            missing_ids = all_ids - used_ids
            
            child.devices_available = [copy.deepcopy(d) for d in self.scenario.devices if d.id in missing_ids]
            
            return child

        c1 = create_child(chromosome1, chromosome2, cut)
        c2 = create_child(chromosome2, chromosome1, cut)
        
        return c1, c2
        

    def mutation(self, chromosome):
        original_chromossome = copy.deepcopy(chromosome)
        op = random.choice(['add','remove', 'move'])

        if op == 'move' and len(chromosome.clusters) >= 1:
            source_cluster = random.choice(chromosome.clusters)
            if len(source_cluster.devices) > 0:
                device_idx = random.randint(0, len(source_cluster.devices) - 1)
                device = source_cluster.devices.pop(device_idx)
                
                if not source_cluster.devices:
                    chromosome.clusters.remove(source_cluster)
                
                if random.random() < 0.5 or not chromosome.clusters:
                    chromosome.devices_available.append(device)
                else:
                    target_cluster = random.choice(chromosome.clusters)
                    target_cluster.devices.append(device)

        if op == 'add' and len(chromosome.clusters) < self.scenario.max_clusters():
            if len(chromosome.devices_available) > 0:
                cluster, du, devices_available = self.generate_random_cluster(chromosome.devices_available)
                if cluster is None:
                    return
                chromosome.clusters.append(cluster)
                chromosome.devices_available = devices_available

        elif op == 'remove' and len(chromosome.clusters) > 1:
            cluster_index = random.randint(0,len(chromosome.clusters)-1)
            chromosome.devices_available.extend(chromosome.clusters[cluster_index].devices)
            del chromosome.clusters[cluster_index]

        if not self.is_valid(chromosome):
            chromosome.clusters = original_chromossome.clusters
            chromosome.devices_available = original_chromossome.devices_available

    def fitness_function(self, chromosome):
        return chromosome.calculate_cost(self.scenario)
        
    def evaluation_function(self, chromosome):
        return self.fitness_function(chromosome)
    
    def best_solution(self):
        solution = None
        min_cost = None
        sum_cost = 0
        for chromosome in self.population:
            cost = self.evaluation_function(chromosome)
            sum_cost += cost
            if  min_cost == None or cost < min_cost:
                min_cost = cost
                solution = chromosome
        avg_cost = sum_cost / len(self.population)
        return solution, min_cost, avg_cost
    
    def is_valid(self, chromosome):
        return True
    
    def tournament(self):
        new_population = []
        tournament_size = 3 
        
        for _ in range(len(self.population) // 2):
            competitors = random.sample(self.population, tournament_size)
            winner = min(competitors, key=lambda ind: ind.cost)
            new_population.append(copy.deepcopy(winner))
            
        return new_population
    
    def generate_new_population(self):
        number_of_crossovers = 0
        number_of_mutations = 0


        parents = self.tournament()  
        offspring = []

        n_parents = len(parents)
        for i in range(0, n_parents, 2):
            p1 = parents[i]
            p2 = parents[(i+1) % n_parents]
            if random.random() < self.crossover_rate:
                number_of_crossovers += 1
                c1, c2 = self.crossover(p1, p2)
            else:
                c1 = copy.deepcopy(p1)
                c2 = copy.deepcopy(p2)

            if random.random() < self.mutation_rate:
                self.mutation(c1)
                number_of_mutations += 1
            if random.random() < self.mutation_rate:
                self.mutation(c2)
                number_of_mutations += 1

            offspring.append(c1)
            offspring.append(c2)

        new_pop = parents + offspring
        if len(new_pop) > self.n_population:
            new_pop = new_pop[:self.n_population]
        else:
            while len(new_pop) < self.n_population:
                new_pop.append(copy.deepcopy(random.choice(parents)))

        self.population = new_pop
        for ind in self.population:
            ind.cost = ind.calculate_cost(self.scenario)

        return number_of_crossovers, number_of_mutations

    def run(self, id):
        gen = []
        avg = []
        solutions = []
        
        for i in range(self.n_gen):
            n_c, n_m = self.generate_new_population()
            best_solution_chromosome, best_solution_cost, average_cost = self.best_solution()
            #logging.info(f"Gen {i+1}: best={best_solution_cost:.10f}, avg={average_cost:.4f}, Pop Size:{len(self.population)}, N_crossovers: {n_c}, N_mutations: {n_m}")
            gen.append(i+1)
            avg.append(average_cost)
            solutions.append(best_solution_cost)

        return (gen, best_solution_chromosome, solutions, avg)
