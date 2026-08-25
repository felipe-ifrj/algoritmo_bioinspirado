import random
import copy
import logging
from network import Network
from cluster import Cluster

logging.basicConfig(level=logging.INFO)

class Particle:
    def __init__(self, position):
        self.position = position
        self.best_position = copy.deepcopy(position)
        self.best_cost = float("inf")

class PSO:

    def __init__(
        self,
        n_particles=10,
        scenario=None,
        n_iter=100,
        w=0.2,        # inércia
        c1=0.95,       # cognitivo
        c2=0.95        # social
    ):
        self.n_particles = n_particles
        self.scenario = scenario
        self.n_iter = n_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.swarm = []
        self.global_best = None
        self.global_best_cost = float("inf")

        self.initialize_swarm()

    # ===============================
    # Inicialização (igual ao AG)
    # ===============================
    def initialize_swarm(self):
        for _ in range(self.n_particles):
            net = self.random_network()
            particle = Particle(net)
            self.swarm.append(particle)

    def random_network(self):
        network = Network(copy.deepcopy(self.scenario.devices))
        devices_available = network.devices_available

        n_clusters = random.randint(1, self.scenario.max_clusters())

        for _ in range(n_clusters):
            if not devices_available:
                break
            cluster, _, devices_available = self.generate_random_cluster(devices_available)
            if cluster:
                network.clusters.append(cluster)

        network.devices_available = devices_available
        return network

    def generate_random_cluster(self, devices_available):
        if not devices_available:
            return None, 0, devices_available

        n = random.randint(1, len(devices_available))
        devices = random.sample(devices_available, n)

        for d in devices:
            devices_available.remove(d)

        return Cluster(copy.deepcopy(devices)), len(devices), devices_available

    # ===============================
    # Avaliação
    # ===============================
    def evaluate(self, particle):
        cost = particle.position.calculate_cost(self.scenario)

        if cost < particle.best_cost:
            particle.best_cost = cost
            particle.best_position = copy.deepcopy(particle.position)

        if cost < self.global_best_cost:
            self.global_best_cost = cost
            self.global_best = copy.deepcopy(particle.position)

    # ===============================
    # Movimento discreto da partícula
    # ===============================
    def move(self, particle):
        new_position = copy.deepcopy(particle.position)

        # Inércia → pequena mutação
        if random.random() < self.w:
            self.local_perturbation(new_position)

        # Cognitivo → aproxima do pbest
        if random.random() < self.c1:
            self.copy_structure(new_position, particle.best_position)

        # Social → aproxima do gbest
        if random.random() < self.c2 and self.global_best:
            self.copy_structure(new_position, self.global_best)

        particle.position = new_position

    # ===============================
    # Operações discretas
    # ===============================
    def local_perturbation(self, network):
        if not network.clusters:
            return

        cluster = random.choice(network.clusters)
        if cluster.devices:
            device = cluster.devices.pop(random.randint(0, len(cluster.devices)-1))
            if not cluster.devices:
                network.clusters.remove(cluster)
            network.devices_available.append(device)

    def copy_structure(self, target, source):
        if not source.clusters:
            return

        cut = random.randint(1, len(source.clusters))
        used_ids = set()

        new_clusters = []
        for cluster in source.clusters[:cut]:
            devices = []
            for d in cluster.devices:
                if d.id not in used_ids:
                    devices.append(copy.deepcopy(d))
                    used_ids.add(d.id)
            if devices:
                new_clusters.append(Cluster(devices))

        all_ids = {d.id for d in self.scenario.devices}
        missing_ids = all_ids - used_ids

        target.clusters = new_clusters
        target.devices_available = [
            copy.deepcopy(d) for d in self.scenario.devices if d.id in missing_ids
        ]

    # ===============================
    # Execução principal
    # ===============================
    def run(self):
        best_costs = []
        avg_costs = []

        for it in range(self.n_iter):
            costs = []

            for particle in self.swarm:
                self.evaluate(particle)
                costs.append(particle.best_cost)

            avg_costs.append(sum(costs) / len(costs))
            best_costs.append(self.global_best_cost)

            for particle in self.swarm:
                self.move(particle)

            # logging.info(f"Iter {it+1}: best={self.global_best_cost:.6f}, avg={avg_costs[-1]:.6f}")

        return self.global_best, best_costs, avg_costs
