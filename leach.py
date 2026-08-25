import random
import copy
import math

from network import Network
from cluster import Cluster


class LEACH:

    # ==========================================
    # Inicialização
    # ==========================================
    def __init__(
        self,
        scenario,
        p=0.3,
        n_rounds=100,
        alpha=0.6,
        beta=0.2,
        gamma=0.2
    ):
        self.scenario = scenario
        self.devices = scenario.devices

        # Parâmetros do LEACH
        self.p = p
        self.n_rounds = n_rounds

        # Pesos da heurística
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Estruturas internas
        self.cluster_heads = []
        self.clusters = {}

    # ==========================================
    # Execução principal
    # ==========================================
    def run(self):

        best_cost = float("inf")
        best_network = None

        for r in range(self.n_rounds):

            # --------------------------------------
            # Seleção dos cluster-heads
            # --------------------------------------
            self._select_cluster_heads()

            # --------------------------------------
            # Formação dos clusters
            # --------------------------------------
            self._assign_nodes_to_clusters()

            # --------------------------------------
            # Conversão para Network
            # --------------------------------------
            network = self._to_network_representation()

            # --------------------------------------
            # Custo específico do LEACH
            # --------------------------------------
            cost = self._calculate_leach_cost(network)

            # --------------------------------------
            # Guarda a melhor solução
            # --------------------------------------
            if cost < best_cost:
                best_cost = cost
                best_network = copy.deepcopy(network)

        return best_cost

    # ==========================================
    # Função de custo do dispositivo
    # ==========================================
    def _calculate_device_cost(self, device):

        """
        Calcula o custo individual do dispositivo.

        Mantém a essência da função original:

            (CPU + Memory + Bandwidth + (1 - Battery)) / 4

        O resultado é limitado ao intervalo [0, +inf],
        evitando custos negativos.
        """

        cpu = max(0.0, float(device.cpu))
        memory = max(0.0, float(device.memory))
        bandwidth = max(0.0, float(device.bandwidth))
        battery = min(
            1.0,
            max(0.0, float(device.battery))
        )

        cost = (
            cpu
            + memory
            + bandwidth
            + (1.0 - battery)
        ) / 4.0

        return max(0.0, cost)

    # ==========================================
    # Seleção dos cluster-heads
    # ==========================================
    def _select_cluster_heads(self):

        active_devices = [
            device
            for device in self.devices
            if device.battery > 0
        ]

        if not active_devices:
            self.cluster_heads = []
            return

        max_battery = max(
            device.battery
            for device in active_devices
        )

        self.cluster_heads = []

        for device in active_devices:

            probability = self.p * (
                device.battery / max_battery
            )

            if random.random() < probability:
                self.cluster_heads.append(device)

        # Garante pelo menos um cluster-head
        if not self.cluster_heads:

            best_device = max(
                active_devices,
                key=lambda device: device.battery
            )

            self.cluster_heads.append(best_device)

    # ==========================================
    # Formação dos clusters
    # ==========================================
    def _assign_nodes_to_clusters(self):

        self.clusters = {
            cluster_head.id: []
            for cluster_head in self.cluster_heads
        }

        if not self.cluster_heads:
            return

        for device in self.devices:

            # Cluster-head não é associado
            # a outro cluster
            if device in self.cluster_heads:
                continue

            # Dispositivo sem bateria é ignorado
            if device.battery <= 0:
                continue

            best_cluster_head = None
            best_score = float("inf")

            for cluster_head in self.cluster_heads:

                # ----------------------------------
                # Distância
                # ----------------------------------
                distance = self._euclidean_distance(
                    device,
                    cluster_head
                )

                # ----------------------------------
                # Tamanho atual do cluster
                # ----------------------------------
                cluster_size = len(
                    self.clusters[cluster_head.id]
                )

                # ----------------------------------
                # Custo do cluster-head
                # ----------------------------------
                cluster_head_cost = (
                    self._calculate_device_cost(
                        cluster_head
                    )
                )

                # ----------------------------------
                # Função heurística do LEACH
                # ----------------------------------
                score = (
                    self.alpha * distance
                    + self.beta * cluster_size
                    + self.gamma * cluster_head_cost
                )

                if score < best_score:

                    best_score = score
                    best_cluster_head = cluster_head

            # --------------------------------------
            # Associação do dispositivo
            # --------------------------------------
            if best_cluster_head is not None:

                self.clusters[
                    best_cluster_head.id
                ].append(device)

    # ==========================================
    # Função de custo do LEACH
    # ==========================================
    def _calculate_leach_cost(self, network):

        """
        Calcula o custo da solução encontrada pelo LEACH.

        A função mantém os componentes da função de
        custo original:

        1. CPU
        2. Memória
        3. Bandwidth
        4. Consumo de bateria

        Além disso, mantém as penalizações existentes
        para dispositivos não utilizados, ausência de
        tipos de sensores e número de clusters.

        O custo final é sempre >= 0.
        """

        if not network.clusters:
            return float("inf")

        total_cost = 0.0

        total_devices = len(
            self.scenario.devices
        )

        used_devices = 0

        # ==========================================
        # Custo dos clusters
        # ==========================================
        for cluster in network.clusters:

            cluster_cost = 0.0
            types_present = set()

            # --------------------------------------
            # Custo dos dispositivos
            # --------------------------------------
            for device in cluster.devices:

                device_cost = (
                    self._calculate_device_cost(
                        device
                    )
                )

                cluster_cost += device_cost

                types_present.add(
                    device.type
                )

            used_devices += len(
                cluster.devices
            )

            # --------------------------------------
            # Penalização por ausência de tipos
            # --------------------------------------
            number_of_types = (
                self.scenario.number_of_sensor_types
            )

            if number_of_types > 0:

                sensor_absence_penalty = (
                    1.0
                    -
                    (
                        len(types_present)
                        /
                        number_of_types
                    )
                )

                sensor_absence_penalty = max(
                    0.0,
                    sensor_absence_penalty
                )

            else:
                sensor_absence_penalty = 0.0

            # --------------------------------------
            # Custo final do cluster
            # --------------------------------------
            cluster_cost *= (
                1.0
                + sensor_absence_penalty
            )

            total_cost += cluster_cost

        # ==========================================
        # Penalização por dispositivos não utilizados
        # ==========================================
        unused_devices = max(
            0,
            total_devices - used_devices
        )

        unused_devices_penalty = (
            unused_devices * 2.0
        )

        # ==========================================
        # Penalização pelo número de clusters
        # ==========================================
        max_clusters = (
            self.scenario.max_clusters()
        )

        missing_clusters = max(
            0,
            max_clusters - len(network.clusters)
        )

        cluster_count_penalty = (
            missing_clusters * 1.5
        )

        # ==========================================
        # Custo total
        # ==========================================
        total_cost += (
            unused_devices_penalty
        )

        total_cost += (
            cluster_count_penalty
        )

        # ==========================================
        # Garantia de custo não negativo
        # ==========================================
        return max(
            0.0,
            total_cost
        )

    # ==========================================
    # Obter coordenadas
    # ==========================================
    def _get_coordinates(self, device):

        # Formato: x / y
        if hasattr(device, "x") and hasattr(device, "y"):
            return device.x, device.y

        # Formato: position_x / position_y
        if (
            hasattr(device, "position_x")
            and hasattr(device, "position_y")
        ):
            return (
                device.position_x,
                device.position_y
            )

        # Formato: coord_x / coord_y
        if (
            hasattr(device, "coord_x")
            and hasattr(device, "coord_y")
        ):
            return (
                device.coord_x,
                device.coord_y
            )

        # Formato: position = (x, y)
        if hasattr(device, "position"):
            return (
                device.position[0],
                device.position[1]
            )

        # Formato: coordinates = (x, y)
        if hasattr(device, "coordinates"):
            return (
                device.coordinates[0],
                device.coordinates[1]
            )

        raise AttributeError(
            f"Device {device.id} has no valid "
            f"coordinate attributes."
        )

    # ==========================================
    # Distância Euclidiana
    # ==========================================
    def _euclidean_distance(
        self,
        device_1,
        device_2
    ):

        x1, y1 = self._get_coordinates(
            device_1
        )

        x2, y2 = self._get_coordinates(
            device_2
        )

        return math.sqrt(
            (x1 - x2) ** 2
            +
            (y1 - y2) ** 2
        )

    # ==========================================
    # Conversão para Network
    # ==========================================
    def _to_network_representation(self):

        network = Network(
            copy.deepcopy(
                self.scenario.devices
            )
        )

        network.clusters = []

        used_ids = set()

        # ------------------------------------------
        # Cria os clusters
        # ------------------------------------------
        for cluster_head in self.cluster_heads:

            cluster_devices = [
                copy.deepcopy(cluster_head)
            ]

            # Adiciona dispositivos associados
            for device in self.clusters.get(
                cluster_head.id,
                []
            ):

                cluster_devices.append(
                    copy.deepcopy(device)
                )

                used_ids.add(
                    device.id
                )

            # Registra o cluster-head
            used_ids.add(
                cluster_head.id
            )

            # Cria o cluster
            network.clusters.append(
                Cluster(cluster_devices)
            )

        # ------------------------------------------
        # Identifica dispositivos não utilizados
        # ------------------------------------------
        all_ids = {
            device.id
            for device in self.scenario.devices
        }

        missing_ids = (
            all_ids - used_ids
        )

        network.devices_available = [
            copy.deepcopy(device)
            for device in self.scenario.devices
            if device.id in missing_ids
        ]

        return network