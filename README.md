# Reduced Order Model of Glioblastoma Growth and its Neuroimaging-informed Estimation of Patient-Specific Parameters

The full process for the parameters estimation consist of the following steps:

# 1. Convert .xml files into .xdmf and get local values for tensors as meshtag
The scripts:
- b_mesh_L.py -> Get mesh and labels
- b_mesh_tend.py -> Get mesh and tensor D value
- b_mesh_tent.py -> Get mesh and tensor T value

are used to create the domain over which Proper Orthogonal Decomposition is performed.

# 2. Create a Reduced Basis 
The scripts:
- pod_ch_brain_offline_splitted.py -> Perform POD over time
- pod_ch_brain_offline_all.py -> Perform POD over parameters

are used to create a reduced basis.

# 3. Create a data-set of pairs parameters-reduced order coefficients
The script:
- pod_ch_brain_online_proj.py -> Compute Reduced Order Solution multiple times

is used to create the data-set.

# 4. Train the neural networks
The script:
- POD_NN_brain.ipynb 

is used to train the neural networks for:
- reduced order coefficients prediction
- model parameters estimation

# 5. Compare different methodologies
The script:
- pod-NN_ch_brain.py

Given a parameters set, is used to compare:
- the Full-Order Model solution
- the Reduced-Order Model solution
- the Reduced-Order Model solution estimated via the reduced order coefficients prediction neural network
- the solution starting from the parameters obtained via model parameters estimation neural network 
