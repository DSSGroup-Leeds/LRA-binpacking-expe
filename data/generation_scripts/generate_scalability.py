import numpy as np
import pandas as pd
from ast import literal_eval
from itertools import product
from random import choices

from graph_utils import create_arbitrary_df, create_normal_df, create_threshold_df

from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent


def load_dataset(filename, isTS):
    df = pd.read_csv(filename, sep='\t', index_col='app_id', encoding='utf-8')
    df.drop(["nb_instances", "inter_degree", "inter_aff"], axis=1, inplace=True)

    if isTS:
        df["core"] = df.core.apply(literal_eval) # Correctly retrieve the list
        df["memory"] = df.memory.apply(literal_eval) # Correctly retrieve the list

    return df


def pick_replicas(n, pop_replicas, wei_replicas):
    return choices(pop_replicas, weights=wei_replicas, k=n)


def create_base_df(nb_apps, size_df, df, pop_replicas, wei_replicas):
    sublist = np.random.randint(low=0, high=size_df, size=(nb_apps,))
    base_df = df.iloc[sublist].copy()
    base_df["app_id"] = range(1, nb_apps+1)
    base_df.set_index("app_id", inplace=True)
    base_df["nb_instances"] = pick_replicas(nb_apps, pop_replicas, wei_replicas)
    return base_df


# Keep replica distribution of the TClab dataset
def get_replicas_distrib(filename):
    df = pd.read_csv(filename, usecols=['nb_instances'], sep='\t')
    pop_replicas = list(df.nb_instances.value_counts().index)
    wei_replicas = df.nb_instances.value_counts().values
    return (pop_replicas, wei_replicas)


def main():
    input_path = BASE_PATH / "TClab"
    output_path = BASE_PATH / "input"

    filename_TS = input_path / "TClab_dataset_TS.csv"

    df_TS = load_dataset(filename_TS, isTS=True)

    pop_replicas, wei_replicas = get_replicas_distrib(filename_TS)

    print("Generating scalability instances for TS")
    generate_from_df(df_TS, output_path/"scalability_TS", pop_replicas, wei_replicas)
# End main function



def generate_from_df(df, output_path, pop_replicas, wei_replicas):
    output_path.mkdir(parents=True, exist_ok=True)

    size_df = len(df)

    list_nb_apps = [10000*x for x in range(1, 11)]

    # We want a density of 0.5%, so a value of 0.005
    densities = {
        "005": 0.005
    }

    cols = ["nb_instances", "core", "memory", "inter_degree", "inter_aff"]

    for nb_apps in list_nb_apps:
        print(f"Generating {nb_apps} apps")

        base_df = create_base_df(nb_apps, size_df, df, pop_replicas, wei_replicas)

        for (s_density, d) in densities.items():
            # Generate Arbitrary graph + df
            df_arbitrary = create_arbitrary_df(nb_apps, d, base_df)
            outfile = output_path/f"scalability_{nb_apps}_arbitrary_d{s_density}.csv"
            df_arbitrary.to_csv(outfile, sep='\t', encoding='utf-8', columns=cols)
            del df_arbitrary # for memory reasons


            # Generate Normal graph + df
            df_normal = create_normal_df(nb_apps, d, base_df)
            outfile = output_path/f"scalability_{nb_apps}_normal_d{s_density}.csv"
            df_normal.to_csv(outfile, sep='\t', encoding='utf-8', columns=cols)
            del df_normal # for memory reasons


            # Generate Threshold graph + df
            df_threshold = create_threshold_df(nb_apps, d, base_df)
            outfile = output_path/f"scalability_{nb_apps}_threshold_d{s_density}.csv"
            df_threshold.to_csv(outfile, sep='\t', encoding='utf-8', columns=cols)
            del df_threshold # for memory reasons

        del base_df # just in case


if __name__ == '__main__':
    main()