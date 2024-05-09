import json
import os
import sys


def experiment_config(save_dir, exp_dir, exp_name, eval_funcs, diversity_funcs, network_size=10, num_generations=1000, 
                      initial_popsize=100, pareto_size=5, mutation_rate=0.01, crossover_rate=0.6):
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
        "initial_popsize": initial_popsize,
        "network_size": network_size,
        "pareto_size": pareto_size,
        "num_generations": num_generations,
        "eval_funcs": eval_funcs,
        "diversity_funcs": diversity_funcs
    }

    config_path = f"{save_dir}{exp_dir}/{exp_name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def initial_performance(exp_dir, network_size, num_generations, initial_popsize, mutation_rate, crossover_rate):
    configs_path = "configs/"
    if not os.path.exists(configs_path+exp_dir):
        os.makedirs(configs_path+exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    eval_funcs = {}
    eval_funcs["c"] = {"connectance": 0.2}
    eval_funcs["pip"] = {"positive_interactions_proportion": 0.25}

    diversity_funcs = {}
    pareto_sizes = {}
    table_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    diversity_funcs["c"] = [{"positive_interactions_proportion": table_vals}, 
                            {"positive_interactions_proportion": table_vals, "average_positive_interactions_strength": table_vals},
                            {"positive_interactions_proportion": table_vals, "clustering_coefficient": table_vals[0:6]}]
    diversity_funcs["pip"] = [{"connectance": table_vals},
                              {"connectance": table_vals, "average_positive_interactions_strength": table_vals},
                              {"connectance": table_vals, "clustering_coefficient": table_vals[0:6]}]
    pareto_sizes["c"] = [22, 2, 4]
    pareto_sizes["pip"] = [22, 2, 4]

    config_names = []
    for exp_name,eval_func in eval_funcs.items():
        for i in range(len(diversity_funcs[exp_name])):
            diversity_func = diversity_funcs[exp_name][i]
            pareto_size = pareto_sizes[exp_name][i]
            exp_namei = exp_name+str(i)
            experiment_config(configs_path, exp_dir, exp_namei, eval_func, diversity_func, network_size, 
                              num_generations, initial_popsize, pareto_size, mutation_rate, crossover_rate)
            config_names.append(exp_namei)
            if not os.path.exists(exp_dir+"/"+exp_namei):
                os.makedirs(exp_dir+"/"+exp_namei)

    submit_output, analysis_output = generate_scripts_batch(exp_dir, config_names)
    return submit_output, analysis_output


def generate_scripts_batch(exp_name, config_names):
    code_location = os.getcwd()+"/"
    configs_path = "configs/"

    submit_output = []
    analysis_output = []
    for config_name in config_names:
        for i in range(0,10):
            submit_output.append(f"python3 graph-evolution/main.py {configs_path}{exp_name}/{config_name}.json {i}\n")
        #submit_output.append(f"sbatch {code_location}run_config_short.sb {configs_path}{exp_name}/{config_name}.json\n")
        analysis_output.append(f"python3 graph-evolution/replicate_analysis.py {exp_name}/{config_name}\n")

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
    if experiment_name == "performance":
        s, a = initial_performance(experiment_name, 10, 1000, 100, 0.01, 0.6)
        submit_output += s
        analysis_output += a
    else:
        print("Invalid experiment name.")
        exit()
    write_scripts_batch(experiment_name, submit_output, analysis_output)