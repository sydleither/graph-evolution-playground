import json
import os
import sys

import numpy as np
from scipy.stats import norm


def experiment_config(save_dir, exp_dir, exp_name, eval_funcs, network_size=10,
                      mutation_rate=0.01, crossover_rate=0.6):
    config = {
        "data_dir": exp_dir,
        "name": exp_name,
        "reps": 1,
        "save_data": 1,
        "plot_data": 0,
        "mutation_rate": mutation_rate,
        "mutation_odds": [1,2,1,2],
        "crossover_odds": [1,2,2],
        "crossover_rate": crossover_rate,
        "weight_range": [-1,1],
        "network_size": network_size,
        "hash_resolution": 100,
        "cell_capacity": 10,
        "num_generations": 1000000,
        "initial_popsize": 10000,
        "eval_funcs": eval_funcs
    }

    config_path = f"{save_dir}{exp_dir}/{exp_name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def parameter_tuning_diversity(exp_dir, network_size, mutation_rate, crossover_rate):
    configs_path = "configs/"
    if not os.path.exists(configs_path+exp_dir):
        os.makedirs(configs_path+exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    eval_funcs = {}
    eval_funcs["c"] = {"connectance": 0.2}
    eval_funcs["ccc"] = {"connectance": 0.2, "clustering_coefficient":0.4}
    eval_funcs["pip"] = {"positive_interactions_proportion": 0.25}
    eval_funcs["pipapis"] = {"positive_interactions_proportion": 0.25, "average_positive_interactions_strength": 0.25}
    eval_funcs["cpip"] = {"connectance": 0.2, "positive_interactions_proportion": 0.25}
    ns_inv = 1/network_size
    dd = [ns_inv*np.round(norm.pdf(x, loc=network_size/4, scale=network_size/10)/ns_inv) for x in range(network_size+1)]
    eval_funcs["dd"] = {"in_degree_distribution": dd}

    config_names = []
    for exp_name,eval_func in eval_funcs.items():
        experiment_config(configs_path, exp_dir, exp_name, eval_func, network_size, mutation_rate, crossover_rate)
        config_names.append(exp_name)
        os.makedirs(exp_dir+"/"+exp_name)

    run_length = "short"
    submit_output, analysis_output = generate_scripts_batch(exp_dir, config_names, run_length)
    return submit_output, analysis_output


def generate_scripts_batch(exp_name, config_names, run_length):
    code_location = os.getcwd()+"/"
    configs_path = "configs/"

    submit_output = []
    analysis_output = []
    for i in range(len(config_names)):
        submit_output.append(f"sbatch {code_location}run_config_{run_length}.sb {configs_path}{exp_name}/{config_names[i]}.json {exp_name}/{config_names[i]}\n")
        analysis_output.append(f"python3 graph-evolution/replicate_analysis.py {exp_name}/{config_names[i]}\n")

    return submit_output, analysis_output


def write_scripts_batch(exp_name, submit_output, analysis_output):
    configs_path = "configs/"

    with open(f"{configs_path}submit_{exp_name}_experiments", "w") as f:
        for output_line in submit_output:
            f.write(output_line)

    with open(f"configs/analyze_{exp_name}_experiments", "w") as f:
        for output_line in analysis_output:
            f.write(output_line)


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    submit_output = []
    analysis_output = []
    if experiment_name == "mapelites":
        s, a = parameter_tuning_diversity(experiment_name, 10, 0.05, 0.6)
        submit_output += s
        analysis_output += a
    else:
        print("Invalid experiment name.")
        exit()
    write_scripts_batch(experiment_name, submit_output, analysis_output)
