# pips-design-toolkit
Computational tools for protein engineering

## A. Set up repository
Clone github repository to local directory: 
```
git clone git@github.com:chicanagram/pips-design-toolkit.git
```
Navigate into the directory:
```
cd pips-design-toolkit
```

## Set up environment

This project uses [Conda](https://docs.conda.io/en/latest/) to manage the Python environment.  
Ensure conda is installed before proceeding. Then create a new conda environment and install packages from requirements.txt
```
conda create -n pips-design-toolkit python=3.11
conda activate pips-design-toolkit
conda install pip
conda install -c conda-forge biopython openprotein
pip install -r requirements.txt
```

## Run notebooks
From within the repo, start Jupyter notebook:
```
jupyter notebook
```
To run a notebook, simply load the notebook and execute the cells in succession. 
Detailed instructions on software requirements and licenses, how to set up the requirements, and input files needed to execute each notebook can be found in the notebooks. 

