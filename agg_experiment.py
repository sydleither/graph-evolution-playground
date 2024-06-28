import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette(sns.color_palette("pastel"))
warnings.filterwarnings("ignore")


def print_info(df, measure):
    best = {x:0 for x in df["exp_name"].unique()}
    for property in sorted(df["property"].unique()):
        dfp = df.loc[(df["property"] == property) & (df["optimized"].isnull())]
        for objectives in dfp["objectives"].unique():
            if property == "connectance" and "dd" in objectives:
                continue
            dfpo = dfp.loc[dfp["objectives"] == objectives]
            dfpo = dfpo.drop(columns=["property", "objectives", "rep", "optimized"])
            dfpo = dfpo.groupby(["exp_name"]).mean()
            best_row = dfpo[dfpo[measure] == dfpo[measure].max()]
            best[best_row.index[0]] += 1
    print(f"{measure} {best}")


def plot_diversity(df, exp_name, measure):
    figure, axis = plt.subplots(4, 4, figsize=(24,24))
    row = 0
    col = 0
    for property in sorted(df["property"].unique()):
        if property.endswith("distribution"):
            continue
        dfp = df.loc[(df["property"] == property) & (df["optimized"].isnull())]
        sns.boxplot(data=dfp, x="objectives", y=measure, hue="exp_name", ax=axis[row][col])
        axis[row][col].set_title(property)
        row += 1
        if row % 4 == 0:
            col += 1
            row = 0
    figure.tight_layout()
    plt.savefig(f"{exp_name}/{measure}.png")
    plt.close()


def get_data(exp_dir):
    df = pd.DataFrame(columns=["exp_name", "objectives", "rep", "property", "entropy", "uniformity", "spread", "optimized"])
    for experiment in os.listdir(exp_dir):
        if os.path.isfile(f"{exp_dir}/{experiment}"):
            continue
        for param_combo in os.listdir(f"{exp_dir}/{experiment}"):
            for replicate in os.listdir(f"{exp_dir}/{experiment}/{param_combo}"):
                full_path = f"{exp_dir}/{experiment}/{param_combo}/{replicate}"
                if os.path.isfile(full_path):
                    continue
                df_i = pd.read_csv(f"{full_path}/diversity.csv")
                df_i["exp_name"] = experiment
                df_i["objectives"] = param_combo
                df_i["rep"] = replicate
                fitnesses = pd.read_pickle(f"{full_path}/fitness_log.pkl")
                fitnesses = {k:v[-1] for k,v in fitnesses.items()}
                for objective in fitnesses.keys():
                    df_i.loc[df_i["property"] == objective, "optimized"] = "yes" if fitnesses[objective] == 0 else "no"
                df = pd.concat([df, df_i])
    return df.reset_index()


def main(exp_name):
    df = get_data(exp_name)
    plot_diversity(df, exp_name, "uniformity")
    plot_diversity(df, exp_name, "spread")
    plot_diversity(df, exp_name, "entropy")
    print_info(df, "spread")
    print_info(df, "uniformity")
    print_info(df, "entropy")


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    main(experiment_name)