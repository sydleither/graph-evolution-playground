import json
import os
import sys


def experiment_config(save_dir, exp_dir, exp_name, eval_funcs, network_size=10, num_generations=1000, popsize=500, mutation_rate=0.01, crossover_rate=0.6):
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
        "popsize": popsize,
        "network_size": network_size,
        "num_generations": num_generations,
        "eval_funcs": eval_funcs
    }

    config_path = f"{save_dir}{exp_dir}/{exp_name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def parameter_tuning_diversity(exp_dir, network_size, num_generations, popsize, mutation_rate, crossover_rate):
    configs_path = "configs/"
    if not os.path.exists(configs_path+exp_dir):
        os.makedirs(configs_path+exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    eval_funcs = {}
    eval_funcs["c25"] = {"connectance": 0.25}
    eval_funcs["c75"] = {"connectance": 0.75}
    eval_funcs["pip"] = {"positive_interactions_proportion": 0.25}
    eval_funcs["pipapis"] = {"positive_interactions_proportion": 0.25, "average_positive_interactions_strength": 0.25}
    eval_funcs["cpip"] = {"connectance": 0.2, "positive_interactions_proportion": 0.25}
    eval_funcs["dd"] = {"in_degree_distribution": [0.0, 0.5, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

    config_names = []
    for exp_name,eval_func in eval_funcs.items():
        experiment_config(configs_path, exp_dir, exp_name, eval_func, network_size, num_generations, popsize, mutation_rate, crossover_rate)
        config_names.append(exp_name)
        os.makedirs(exp_dir+"/"+exp_name)

    submit_output, analysis_output = generate_scripts_batch(exp_dir, config_names)
    return submit_output, analysis_output


def generate_scripts_batch(exp_name, config_names):
    code_location = os.getcwd()+"/"
    configs_path = "configs/"

    submit_output = []
    analysis_output = []
    for config_name in config_names:
        submit_output.append(f"sbatch {code_location}run_config.sb {configs_path}{exp_name}/{config_name}.json {exp_name}\n")
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
    if experiment_name == "diversity":
        for numgen in [500, 1000]:
            for popsize in [250, 500]:
                s, a = parameter_tuning_diversity(experiment_name+f"_{numgen}gen_{popsize}pop", 10, numgen, popsize, 0.005, 0.6)
                submit_output += s
                analysis_output += a
    elif experiment_name == "mutation":
        mutation_rates = [0.01, 0.05, 0.1]
        file_names = [1, 2, 10]
        for i in range(len(mutation_rates)):
            s, a = parameter_tuning_diversity(experiment_name+f"_{file_names[i]}", 10, 1000, 500, mutation_rates[i], 0.6)
            submit_output += s
            analysis_output += a
    elif experiment_name == "crossover":
        crossover_rates = [0.4, 0.5, 0.6, 0.7, 0.8]
        for cr in crossover_rates:
            s, a = parameter_tuning_diversity(experiment_name+f"_{str(cr)[-1]}", 10, 1000, 500, 0.01, cr)
            submit_output += s
            analysis_output += a
    else:
        print("Invalid experiment name.")
        exit()
    write_scripts_batch(experiment_name, submit_output, analysis_output)