import pandas as pd
from ast import literal_eval
from graph_utils import create_arbitrary_df, create_normal_df, create_threshold_df

from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent

def load_dataset(filename, isTS):
    df = pd.read_csv(filename, sep='\t', index_col='app_id', encoding='utf-8')
    df.drop(["inter_degree", "inter_aff"], axis=1, inplace=True)

    if isTS:
        df["core"] = df.core.apply(literal_eval) # Correctly retrieve the list
        df["memory"] = df.memory.apply(literal_eval) # Correctly retrieve the list

    return df


def main():
    input_path = BASE_PATH / "TClab"
    output_path = BASE_PATH / "input"
    
    filename_2D = input_path / "TClab_dataset_2D.csv"
    filename_TS = input_path / "TClab_dataset_TS.csv"

    print("Generating high degree instances for 2D")
    base_df = load_dataset(filename_2D, isTS=False)
    generate_from_df(base_df, output_path/"density2D")

    print("Generating high degree instances for TS")
    base_df = load_dataset(filename_TS, isTS=True)
    generate_from_df(base_df, output_path/"densityTS")
# End main function


def generate_from_df(base_df, output_path):
    output_path.mkdir(parents=True, exist_ok=True)

    densities = [1, 5, 10]
    instances = range(10)

    nb_apps = len(base_df.index)
    cols = ['nb_instances', 'core', 'memory', 'inter_degree', 'inter_aff']

    for int_d in densities:
        d = int_d / 100.0
        print(f"Density {d}%: ")

        s = ""
        for i in instances:
            s+= f"{i} "
            print(f"\r{s}", end='')
            # Generate Arbitrary graph + df
            df_arbitrary = create_arbitrary_df(nb_apps, d, base_df)
            outfile = output_path/f"arbitrary_d{int_d}_{i}.csv"
            df_arbitrary.to_csv(outfile, sep='\t', encoding='utf-8', columns=cols)
            del df_arbitrary # for memory reasons

            # Generate Normal graph + df
            df_normal = create_normal_df(nb_apps, d, base_df)
            outfile = output_path/f"normal_d{int_d}_{i}.csv"
            df_normal.to_csv(outfile, sep='\t', encoding='utf-8', columns=cols)
            del df_normal # for memory reasons

            # Generate Threshold graph + df
            df_threshold = create_threshold_df(nb_apps, d, base_df)
            outfile = output_path/f"threshold_d{int_d}_{i}.csv"
            df_threshold.to_csv(outfile, sep='\t', encoding='utf-8', columns=cols)
            del df_threshold # for memory reasons
        print("")


if __name__ == '__main__':
    main()
