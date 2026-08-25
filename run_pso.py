import pandas as pd
import numpy as np
import logging
import os
import time
import matplotlib.pyplot as plt

from device import Device
from scenario import Scenario
from pso import PSO


# ======================================
# Configuração de Log
# ======================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ======================================
# Função para carregar cenários
# ======================================
def load_scenarios(n_scenarios):

    scenarios = {}

    for t in range(n_scenarios):
        devices = []

        file_name = f"dataset_transformed/csv_files/t_{t}.csv"
        df = pd.read_csv(file_name)

        for row in df.itertuples(index=False):
            devices.append(
                Device(
                    type=int(row.type),
                    id=int(row.id),
                    cpu=float(row.cpu),
                    memory=float(row.memory),
                    battery=float(row.battery),
                    bandwidth=float(row.bandwidth)
                )
            )

        scenarios[t] = Scenario(
            devices=devices,
            number_of_sensor_types=9,
            number_of_devices_per_type=10
        )

    return scenarios


# ======================================
# Main
# ======================================
if __name__ == "__main__":

    # =======================
    # Parâmetros do PSO
    # =======================
    n_particles = 100
    n_iter = 100
    n_runs = 100
    n_scenarios = 9

    w = 0.2
    c1 = 0.95
    c2 = 0.95
    
    
    #w = 0.25 original = mutação(no ga)
    #c1 = 0.70 original = crossover(no ga)
    #c2 = 0.70 original 
    

    os.makedirs("results", exist_ok=True)

    # Estrutura de armazenamento
    results = {
        t: {"cost": [], "time": []}
        for t in range(n_scenarios)
    }

    # -----------------------
    # Carregamento cenários
    # -----------------------
    scenarios = load_scenarios(n_scenarios)

    # -----------------------
    # Execuções do PSO
    # -----------------------
    for run in range(n_runs):
        logging.info(f"================ RUN {run + 1}/{n_runs} ================")

        for t_index, scenario in scenarios.items():

            start = time.perf_counter()

            pso = PSO(
                n_particles=n_particles,
                scenario=scenario,
                n_iter=n_iter,
                w=w,
                c1=c1,
                c2=c2
            )

            best_solution, best_costs, avg_costs = pso.run()

            end = time.perf_counter()

            exec_time = end - start

            # Armazena melhor custo final
            results[t_index]["cost"].append(best_costs[-1])
            results[t_index]["time"].append(exec_time)

        logging.info(f"Run {run + 1} finished")

    # -----------------------
    # Estatísticas
    # -----------------------
    scenarios_idx = list(range(1, n_scenarios + 1))

    mean_costs, std_costs = [], []
    mean_times, std_times = [], []

    for t in range(n_scenarios):

        cost_values = np.array(results[t]["cost"])
        time_values = np.array(results[t]["time"])

        mean_costs.append(cost_values.mean())
        std_costs.append(cost_values.std(ddof=1))

        mean_times.append(time_values.mean())
        std_times.append(time_values.std(ddof=1))

    # -----------------------
    # CSV consolidado
    # -----------------------
    stats_df = pd.DataFrame({
        "scenario": scenarios_idx,
        "mean_cost": mean_costs,
        "std_cost": std_costs,
        "mean_execution_time_sec": mean_times,
        "std_execution_time_sec": std_times
    })

    stats_df.to_csv("results/statistical_summary_pso.csv", index=False)

    # -----------------------
    # CSV apenas tempo
    # -----------------------
    time_df = pd.DataFrame({
        "scenario": scenarios_idx,
        "mean_execution_time_sec": mean_times,
        "std_execution_time_sec": std_times
    })

    time_df.to_csv(
        "results/statistical_summary_execution_time_pso.csv",
        index=False
    )

    # -----------------------
    # Gráfico – Custo
    # -----------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        scenarios_idx,
        mean_costs,
        marker="o",
        color="green",
        label="Mean Best Cost (PSO)"
    )

    plt.fill_between(
        scenarios_idx,
        np.array(mean_costs) - np.array(std_costs),
        np.array(mean_costs) + np.array(std_costs),
        alpha=0.3
    )

    plt.xlabel("Scenario Index")
    plt.ylabel("Best Cost")
    plt.title("PSO Performance (Mean ± Std Dev)")
    plt.grid(True)
    plt.legend()

    plt.savefig("results/mean_std_behavior_overtime_pso.png")
    plt.close()

    # -----------------------
    # Gráfico – Tempo
    # -----------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        scenarios_idx,
        mean_times,
        marker="o",
        color="blue",
        label="Mean Execution Time (PSO)"
    )

    plt.fill_between(
        scenarios_idx,
        np.array(mean_times) - np.array(std_times),
        np.array(mean_times) + np.array(std_times),
        alpha=0.3
    )

    plt.xlabel("Scenario Index")
    plt.ylabel("Execution Time (seconds)")
    plt.title("PSO Execution Time (Mean ± Std Dev)")
    plt.grid(True)
    plt.legend()

    plt.savefig("results/mean_std_execution_time_pso.png")
    plt.close()

    logging.info("PSO processing completed successfully.")