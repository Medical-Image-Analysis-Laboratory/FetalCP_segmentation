# Author: Priscille de Dumast
# Date: 15.08.2022

import os
import json
import pandas as pd
import numpy as np

from topofetal.pipelines.infer_feta20_pipeline import InferencePipeline

# ignore all future warnings (scikit-learn)
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)

# Fix random
np.random.seed(1312)


def main():

    code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(os.path.join(code_dir, "configs", "config_paths.json")) as jsonFile:
        config_paths = json.load(jsonFile)
        jsonFile.close()

    networks_dir = config_paths["networks_dir"]
    results_dir = config_paths["results_dir"]
    feta20_dir = config_paths["feta20_dir"]

    participants = pd.read_csv(os.path.join(code_dir, 'configs', 'participants_feta20.csv'))
    subjects_testing  = participants[participants['Fold'].isin([-1])]['participant_id']

    for sub in subjects_testing:
        for config in ["Baseline","Hybrid","TopoCP"]:
            pip = InferencePipeline(
                p_data_dir=feta20_dir,
                p_output_dir=os.path.join(results_dir),
                p_subject=sub,
                p_net_dir=networks_dir,
                p_net_name=config
            )
            pip.create_workflow()
            pip.run()
            break
        break


if __name__ == '__main__':

    main()