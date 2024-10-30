# *Usage of CSformer to analysis MDSimulation Trajectories*

****

## Step 1  Data Process
Process the trajectory (xtc files) using featurization.py

## Step 2 Model Training
Train the model by running train.py, and modify the configuration in config first.

## Step 3 Importance analysis by gradient
Calculate the gradient using integrated_gradient.py and visualize the result using explanation_from_gardient.py
