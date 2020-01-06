# Input Convex Neural Networks and Deep-Q-Learning 
## Project Structure
```bash
├───configs # .yml setup file for anaconda environment 
├───src
│   └───core
        └───simulation_main.py  # Configure and run simulation parameters
        └───simulation_evaluation_main.ipynb  # View simulation results    
│       ├───layers # Contains PICNN & ICNN model architecture
│       ├───optimization # Contains algorithms Bundle Entropy, Projected Newton & PDIPM
│       ├───simulation
            └───simulation.py  # Deep-Q-Learning algorithm
            # Contains further simulation architecture components 
│       └───utils # Contains various helper functions
├───simulation_data # Data with the 3 examples for the jupyter notebooks
```

## Simulation Setup
#### 1. Clone repository

Follow https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository on how to clone the remote repository to local machine 

#### 2. Create Anaconda Environment 

Follow https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html on how to create a anaconda environment from the provided .yml file in python 3.7

#### 3. Adjust simulation parameters

In src/core/simulation_main.py adjust simulation parameters 

Note: Change the log path to store the simulation data 

#### 4. Run sumulation

Save changes in simulation_main.py and run the script in conda environment

#### 5. View Results

Load log data in simulation_evaluation_main.ipynb Jupyter Notebook and view the simulation data
