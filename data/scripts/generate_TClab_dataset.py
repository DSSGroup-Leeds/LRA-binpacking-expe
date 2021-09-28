import numpy as np
import pandas as pd
from math import ceil
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent


def main():
    TClab_dir = BASE_PATH / "TClab"

    # Retrieve resource usage of applications
    app_resources_filename = TClab_dir / "app_resources.csv"
    header = ["app_id", "cpu_usage", "mem_usage"]

    df_resources = pd.read_csv(app_resources_filename, names=header, usecols=[0,1,2])
    df_resources.app_id = df_resources.app_id.apply(lambda x: int(x.split('_')[1]))
    df_resources["cpu_usage_TS"] = df_resources.cpu_usage.apply(lambda x: [ float(i) for i in x.split('|')])
    df_resources["mem_usage_TS"] = df_resources.mem_usage.apply(lambda x: [ float(i) for i in x.split('|')])
    df_resources.set_index("app_id", inplace=True)

    # For the 2D fixed resource requirement
    df_resources["core"] = df_resources.cpu_usage_TS.apply(max).apply(ceil)
    df_resources["memory"] = df_resources.mem_usage_TS.apply(max).apply(ceil)

    # Retrieve the number of replicas
    instance_deployment_filename = TClab_dir / "instance_deployment.csv"
    df_instances = pd.read_csv(instance_deployment_filename, names=['inst_id', 'app_id', 'machine_id'])
    df_instances['app_id'] = df_instances['app_id'].apply(lambda x: int(x.split('_')[1]))
    df_instances.drop(['machine_id'], axis=1, inplace=True)

    df_resources["nb_instances"] = df_instances.groupby(['app_id']).aggregate(lambda x: len(x))


    # Compute affinity relations
    affinity_filename = TClab_dir / "app_interference.csv"
    df_interference = pd.read_csv(affinity_filename, names=['app_a', 'app_b', 'k'])

    # Remove the 'app_' and turn every app id into int
    df_interference.app_a = df_interference.app_a.apply(lambda x: int(x.split('_')[1]))
    df_interference.app_b = df_interference.app_b.apply(lambda x: int(x.split('_')[1]))

    app_list = list(df_interference.app_a.unique())
    df_inter = df_interference[df_interference.app_a != df_interference.app_b]
    app_inter = np.union1d(df_inter.app_a.unique(), df_inter.app_b.unique())
    
    aff_dict = {}

    for app_id in app_list:
        d = {}
        if app_id in app_inter:
            a = list(df_inter[df_inter.app_a == app_id]["app_b"])
            b = list(df_inter[df_inter.app_a == app_id]["k"])
            d["inter_aff"] = list(zip(a,b))
        else:
            d["inter_aff"] = []
        d["inter_degree"] = len(d["inter_aff"])

        aff_dict[app_id] = d

    df_affinity = pd.DataFrame.from_dict(aff_dict, orient='index')
    df_final = df_resources.join(df_affinity)

    # Store the dataset for fixed 2D requirement
    outfile_2D = TClab_dir / "TClab_dataset_2D.csv"
    columns = ['nb_instances', 'core', 'memory', 'inter_degree', 'inter_aff']
    df_final.to_csv(outfile_2D, sep='\t', columns=columns, encoding='utf-8')

    # Store the dataset for time-varying requirement
    outfile_TS = TClab_dir / "TClab_dataset_TS.csv"
    df_final.drop(['core', 'memory'], axis=1, inplace=True)
    df_final.rename(columns={'cpu_usage_TS':'core', 'mem_usage_TS':'memory'}, inplace=True)
    df_final.to_csv(outfile_TS, sep='\t', columns=columns, encoding='utf-8')
# End main function

if __name__ == '__main__':
    main()
