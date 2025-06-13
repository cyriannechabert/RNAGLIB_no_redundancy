"""experiment setup."""

import os
import sys
import shutil

import torch

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import get_task, RNA_CM
from rnaglib.transforms import GraphRepresentation, RNAFMTransform
from rnaglib.dataset_transforms import CDHitComputer, ClusterSplitter, StructureDistanceComputer, RandomSplitter
from rnaglib.encoders import ListEncoder
from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from exp import RNATrainer

# Hyperparameters (to tune)
nb_layers = 2 
hidden_dim = 64 
learning_rate = 0.0005 
batch_size = 16 
epochs = 80 
split = "default" 
rna_fm = False 
representation = "2.5D" 
layer_type = "rgcn" 
output = "tensorboard"  

# Experiment name
exp_name="RNA_CM_"+str(nb_layers)+"layers_lr"+str(learning_rate)+"_"+str(epochs)+"epochs_hiddendim"+str(hidden_dim)+"_"+representation+"_layer_type_"+layer_type
if rna_fm:
    exp_name += "rna_fm"
if split != "default":
     exp_name += split


model_args = {
    "graph_level": False,
    "num_layers": nb_layers,
    "hidden_channels": hidden_dim,
    #"layer_type": layer_type,
}

if rna_fm:
    model_args["num_node_features"]=644

#model_CM = PygModel(**model_args)
#trainer_CM = RNATrainer(ta, model_CM, rep, exp_name=exp_name, learning_rate=learning_rate, epochs=epochs)
#trainer_CM.train()

if __name__ == "__main__":
    for seed in [0,1,2]:
        ta = get_task(root="roots/RNA_CM", task_id="rna_cm")
        if split=="struc":
            distance = "USalign"
        else:
                distance = "cd_hit"

        if distance not in ta.dataset.distances:
            if split == 'struc':
                    ta.dataset = StructureDistanceComputer()(ta.dataset)
            if split == 'seq':
                    ta.dataset = CDHitComputer()(ta.dataset)
        if split == 'rand':
            ta.splitter = RandomSplitter()
        elif split=='struc' or split=='seq':
            ta.splitter = ClusterSplitter(distance_name=distance)

        if rna_fm:
            rnafm = RNAFMTransform()
            [rnafm(rna) for rna in ta.dataset]
            ta.dataset.features_computer.add_feature(feature_names=["rnafm"], custom_encoders={"rnafm": ListEncoder(640)})

        if representation=="2D":
            edge_map = GRAPH_KEYS["2D_edge_map"][TOOL]
        elif representation=="simplified_2.5D":
            edge_map = GRAPH_KEYS["simplified_edge_map"][TOOL]
        else:
            edge_map = GRAPH_KEYS["edge_map"][TOOL]

        representation_args = {
            "framework": "pyg",
            "edge_map": edge_map,
        }

        rep = GraphRepresentation(**representation_args)
        ta.dataset.add_representation(rep)
        ta.get_split_loaders(batch_size=batch_size, recompute=True)
        model = PygModel.from_task(ta, **model_args)
        trainer = RNATrainer(ta, model, rep, exp_name=exp_name+"_seed"+str(seed), learning_rate=learning_rate, epochs=epochs, seed=seed, batch_size=batch_size, output=output)
        trainer.train()