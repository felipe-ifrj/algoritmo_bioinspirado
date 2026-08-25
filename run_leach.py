import pandas as pd
import numpy as np
import logging
import os
import time
import matplotlib.pyplot as plt
import random

from device import Device
from scenario import Scenario
from leach import LEACH


# ===============================
# Configuração de Log
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ===============================
# Função para carregar cenários
# ===============================
def load_scenarios(n_scenarios):

    scenarios = {}

    for t in range(n_scenarios):

        devices = []

        df = pd.read_csv(
            f"dataset_transformed/csv_files/t_{t}.csv"
        )

        print(f"Number of devices simulated: {len(df)}")

        for row in df.itertuples(index=False):

            # -----------------------------------
            # Coordenadas
            # -----------------------------------
            if "x" in df.columns and "y" in df.columns:
                x = float(row.x)
                y = float(row.y)

            elif "position_x" in df.columns and "position_y" in df.columns:
                x = float(row.position_x)
                y = float(row.position_y)

            elif "coord_x" in df.columns and "coord_y" in df.columns:
                x = float(row.coord_x)
                y = float(row.coord_y)

            else:
                # fallback caso dataset não tenha coordenadas
                x = random.uniform(0, 100)
                y = random.uniform(0, 100)

            device = Device(
                type=int(row.type),
                id=int(row.id),
                cpu=float(row.cpu),
                memory=float(row.memory),
                battery=float(row.battery),
                bandwidth=float(row.bandwidth)
            )

            # adiciona coordenadas dinamicamente
            device.x = x
            device.y = y

            devices.append(device)

        scenarios[t] = Scenario(
            devices=devices,
            number_of_sensor_types=9,
            number_of_devices_per_type=10
        )

    return scenarios


# ===============================
# Função Principal
# ===============================
if __name__ == "__main__":

    n_runs = 100
    n_scenarios = 9

    os.makedirs("results", exist_ok=True)

    results = {
        t: {
            "cost": [],
            "time": []
        }
        for t in range(n_scenarios)
    }

    # -------------------------------
    # Carrega cenários
    # -------------------------------
    scenarios = load_scenarios(n_scenarios)

    # -------------------------------
    # Execuções do LEACH
    # -------------------------------
    for run in range(n_runs):

        logging.info(
            f"RUN {run + 1}/{n_runs}"
        )

        for t, scenario in scenarios.items():

            start = time.perf_counter()

            leach = LEACH(
                scenario=scenario,
                p=0.3,
                n_rounds=100
            )

            cost = leach.run()
            print(f"\nScenario {t}") # teste
            print(f"Returned cost = {cost}")  #teste

            end = time.perf_counter()

            exec_time = end - start

            results[t]["cost"].append(cost)
            results[t]["time"].append(exec_time)

    # -------------------------------
    # Estatísticas
    # -------------------------------
    scenarios_idx = list(
        range(1, n_scenarios + 1)
    )

    mean_costs = []
    std_costs = []

    mean_times = []
    std_times = []

    for t in range(n_scenarios):

        cost_values = np.array(
            results[t]["cost"]
        )

        time_values = np.array(
            results[t]["time"]
        )

        mean_costs.append(
            cost_values.mean()
        )

        std_costs.append(
            cost_values.std(ddof=1)
        )

        mean_times.append(
            time_values.mean()
        )

        std_times.append(
            time_values.std(ddof=1)
        )

    # -------------------------------
    # CSV consolidado
    # -------------------------------
    df_stats = pd.DataFrame({
        "scenario": scenarios_idx,
        "mean_cost": mean_costs,
        "std_cost": std_costs,
        "mean_execution_time_sec": mean_times,
        "std_execution_time_sec": std_times
    })

    df_stats.to_csv(
        "results/statistical_summary_leach.csv",
        index=False
    )


        # -----------------------
    # CSV apenas tempo
    # -----------------------
    time_df = pd.DataFrame({
        "scenario": scenarios_idx,
        "mean_execution_time_sec": mean_times,
        "std_execution_time_sec": std_times
    })

    time_df.to_csv(
        "results/statistical_summary_execution_time_leach.csv",
        index=False
    )


    # -------------------------------
    # Gráfico custo
    # -------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        scenarios_idx,
        mean_costs,
        marker="o",
        color="orange",
        label="Mean Energy Cost (LEACH-MA)"
    )

    plt.fill_between(
        scenarios_idx,
        np.array(mean_costs) - np.array(std_costs),
        np.array(mean_costs) + np.array(std_costs),
        alpha=0.4
    )

    plt.xlabel("Scenario Index")
    plt.ylabel("Energy Cost")
    plt.title("LEACH Performance")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        "results/mean_std_behavior_overtime_leach.png"
    )

    plt.close()

    # -------------------------------
    # Gráfico tempo
    # -------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        scenarios_idx,
        mean_times,
        marker="o",
        color="blue",
        label="Mean Execution Time"
    )

    plt.fill_between(
        scenarios_idx,
        np.array(mean_times) - np.array(std_times),
        np.array(mean_times) + np.array(std_times),
        alpha=0.3
    )

    plt.xlabel("Scenario Index")
    plt.ylabel("Execution Time (seconds)")
    plt.title("LEACH Execution Time")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        "results/mean_std_execution_time_leach.png"
    )

    plt.close()

    logging.info(
        "Processamento finalizado com sucesso."
    )