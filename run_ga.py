import pandas as pd
import numpy as np
import logging
import os
import time
import matplotlib.pyplot as plt

from device import Device
from scenario import Scenario
from ga import GA


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_scenarios(n_scenarios):

    scenarios = {}

    for t in range(n_scenarios):

        df = pd.read_csv(f"dataset_transformed/csv_files/t_{t}.csv")

        devices = [
            Device(
                type=int(row.type),
                id=int(row.id),
                cpu=float(row.cpu),
                memory=float(row.memory),
                battery=float(row.battery),
                bandwidth=float(row.bandwidth)
            )
            for row in df.itertuples(index=False)
        ]

        scenarios[t] = Scenario(
            devices=devices,
            number_of_sensor_types=9,
            number_of_devices_per_type=10
        )

    return scenarios


if __name__ == "__main__":

    # =======================
    # Parâmetros GA
    # =======================
    #mutation_rate = 0.25 original
    #crossover_rate = 0.70 original
    mutation_rate = 0.3
    crossover_rate = 0.3
    n_generations = 100
    population_size = 100
    n_runs = 100
    n_scenarios = 9

    os.makedirs("results", exist_ok=True)

    scenarios = load_scenarios(n_scenarios)

    results = {t: {"cost": [], "time": []} for t in range(n_scenarios)}

    # =======================
    # Execuções
    # =======================
    for run in range(n_runs):

        logging.info(f"================ RUN {run + 1}/{n_runs} ================")

        for t_index, scenario in scenarios.items():

            start = time.perf_counter()

            ga = GA(
                n_population=population_size,
                scenario=scenario,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                n_gen=n_generations
            )

            gen, best_solution, best_costs, avg_costs = ga.run(t_index)

            end = time.perf_counter()

            execution_time = end - start

            # melhor custo final da última geração
            final_cost = best_costs[-1]

            results[t_index]["cost"].append(final_cost)
            results[t_index]["time"].append(execution_time)

    # =======================
    # Estatísticas
    # =======================
    scenarios_idx = list(range(1, n_scenarios + 1))

    mean_costs = []
    std_costs = []
    mean_times = []
    std_times = []

    for t in range(n_scenarios):

        cost_array = np.array(results[t]["cost"])
        time_array = np.array(results[t]["time"])

        mean_costs.append(cost_array.mean())
        std_costs.append(cost_array.std(ddof=1))

        mean_times.append(time_array.mean())
        std_times.append(time_array.std(ddof=1))
        
        
        

    # =======================
    # CSV Consolidado
    # =======================
    stats_df = pd.DataFrame({
        "scenario": scenarios_idx,
        "mean_cost": mean_costs,
        "std_cost": std_costs,
        "mean_execution_time_sec": mean_times,
        "std_execution_time_sec": std_times
    })

    stats_df.to_csv("results/statistical_summary_ga.csv", index=False)
    
        # -------------------------------
    # CSV apenas tempo
    # -------------------------------
    df_time = pd.DataFrame({
        "scenario": scenarios_idx,
        "mean_execution_time_sec": mean_times,
        "std_execution_time_sec": std_times
    })

    df_time.to_csv(
        "results/statistical_summary_execution_time_ga.csv",
        index=False
    )

    
    

    # =======================
    # Gráfico – Custo
    # =======================
    plt.figure(figsize=(10, 6))

    plt.plot(
        scenarios_idx,
        mean_costs,
        marker="o",
        color="red",
        label="Mean Best Cost (GA)"
    )

    plt.fill_between(
        scenarios_idx,
        np.array(mean_costs) - np.array(std_costs),
        np.array(mean_costs) + np.array(std_costs),
        alpha=0.3
    )

    plt.xlabel("Scenario Index")
    plt.ylabel("Best Cost")
    plt.title("GA Performance (Mean ± Std Dev)")
    plt.grid(True)
    plt.legend()

    plt.savefig("results/mean_std_behavior_overtime_ga.png")
    plt.close()

    # =======================
    # Gráfico – Tempo
    # =======================
    plt.figure(figsize=(10, 6))

    plt.plot(
        scenarios_idx,
        mean_times,
        marker="o",
        color="blue",
        label="Mean Execution Time (GA)"
    )

    plt.fill_between(
        scenarios_idx,
        np.array(mean_times) - np.array(std_times),
        np.array(mean_times) + np.array(std_times),
        alpha=0.3
    )

    plt.xlabel("Scenario Index")
    plt.ylabel("Execution Time (seconds)")
    plt.title("GA Execution Time (Mean ± Std Dev)")
    plt.grid(True)
    plt.legend()

    plt.savefig("results/mean_std_execution_time_ga.png")
    plt.close()

    logging.info("GA processing completed successfully.")