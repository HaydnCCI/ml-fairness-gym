import logging
import os
import sys
import pandas as pd
import torch
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from source.losses import preference_loss_function
from source.mlp import MLP
from source.training import train_reward_model

# trajectory_path = "./data/synthetic_decisions.csv"
# trajectory_path = "./data/trajectory_decisions.csv"

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)

def train_model(trajectory_path, save_path="./models/reward_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    synthetic_decisions_dataframe = pd.read_csv(trajectory_path)
    features = synthetic_decisions_dataframe.drop("decision", axis=1).to_numpy()
    decisions = synthetic_decisions_dataframe["decision"].to_numpy()
    input_dim = features.shape[1]

    reward_model = MLP(name="reward_model", layer_dims=[input_dim+1, 100, 100, 1], out_act=None)
    losses = train_reward_model(
        reward_model,
        features,
        decisions,
        loss_function=preference_loss_function,
        learning_rate=0.0001,
        num_epochs=50,
        batch_size=256,
        save_path=save_path,
    )

