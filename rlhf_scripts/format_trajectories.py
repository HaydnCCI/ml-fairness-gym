
import ast
import json
import pandas as pd

simulation_results_path = "./data/500_steps/sims_results.txt"
decision_data_path = "./data/trajectory_decisions.csv"

with open(simulation_results_path, 'r') as f:
    data = f.readlines()
    
trajectories = " ".join(data).replace('\n', ' ').split("'trajectories':")[1].strip().strip('}')
trajectories_formatted = ast.literal_eval(trajectories)
trajectories_formatted_df = pd.DataFrame(trajectories_formatted)

X = pd.DataFrame(list(map(list, trajectories_formatted_df[0].values)))
y = trajectories_formatted_df[[1]]
y.columns = ['decision']
df = pd.concat([X, y], axis=1)

df.to_csv(decision_data_path, index=False)
