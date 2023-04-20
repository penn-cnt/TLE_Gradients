# Repository for analyses related to the manuscript: "Subcortical Functional Connectivity Gradients in Temporal Lobe Epilepsy"

For the full manuscript see: https://doi.org/10.1101/2023.01.08.23284313

## Running the analyses

We provide a stand-alone Jupyter notebook as a companion to the manuscript. The notebook runs all the analyses required to generate figures and findings in the manuscript.

### Installing the Conda virtual environment

Please make sure you have [Anaconda](https://www.anaconda.com/) installed. After cloning, open the repository folder in a terminal window and type: 
```
conda env create -f environment.yml
```
this will create a new Python virtual environment with all the required libraries

Activate the environment with: 
```
conda activate tle_gradients
```

### Running the Jupyter notebook

In order to run the analysis, clone this repository and install the conda virtual environment. Open the Jupyter Notebook `manuscript_companion.ipynb` and execute the analyses from there. If the conda environment was setup appropriately and the connectivity matrices were placed in the correct folder, everything should run correctly.
