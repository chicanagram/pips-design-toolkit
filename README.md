# pips-design-toolkit
Computational tools for protein engineering

## 1. Set up repository
Clone github repository to local directory: 
```
git clone git@github.com:chicanagram/pips-design-toolkit.git
```
Navigate into the directory:
```
cd pips-design-toolkit
```

## 2. Set up environment

This project uses [Conda](https://docs.conda.io/en/latest/) to manage the Python environment.
Ensure conda is installed before proceeding.

To rebuild the environment from scratch, first remove the existing environment if it already exists:
```
conda deactivate
conda env remove -n pips-design-toolkit -y
```

AutoGluon on macOS uses LightGBM, which depends on `libomp` for multi-threading.
Install `libomp` before setting up the Python environment:
```
brew uninstall -f libomp
brew install libomp
```

Then create a fresh conda environment and install AutoGluon first:
```
conda create -n pips-design-toolkit python=3.11 -y
conda activate pips-design-toolkit
python -m pip install -U pip wheel
python -m pip install "setuptools<81"
python -m pip install autogluon==1.1.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

After AutoGluon is installed successfully, install the remaining repo dependencies from `requirements.txt`:
```
pip install -r requirements.txt
```

To enable the optional AutoGluon XGBoost and FastAI model families used by the evaluation script, install:
```
python -m pip install "autogluon.tabular[xgboost,fastai]==1.1.1"
```

Do not separately run `conda install numpy scipy scikit-learn` in this environment afterward, as mixing compiled `conda` and `pip` builds can lead to NumPy / SciPy binary incompatibility errors.
Also keep `setuptools<81` and `psutil<6`, since AutoGluon 1.1.1 still depends on `pkg_resources` and requires an older `psutil` version.
## 3. Usage
The code can be run directly via python scripts (see section A), through command line an IDE, or via Jupyter notebooks  (see section B). The list of scripts and notebooks available is found in section C. 

### A) Run code from scripts
To run a script, first navigate to the directory in which the script is located. 
```
cd tools
```
**Prior to running the script, edit the inputs required for that particular script in the `__main__` section right at the bottom of in the script.** Descriptions of each input can be found in the corresponding notebook. 

Then, to run a script (e.g. `./tools/compile_analyse_dataset.py`) from command line, execute: 
```
python compile_analyse_dataset.py
```

### B) Run code from jupyter notebooks
From within the repo, start Jupyter notebook:
```
jupyter notebook notebooks/
```
To run a notebook, simply load the notebook and execute the cells in succession. 
Detailed instructions on software requirements and licenses, how to set up the requirements, and input files needed to execute each notebook can be found in the notebooks.

### C) List of scripts / notebooks:
#### 1. Get conservation and distance scores

Computes residue-level conservation scores from multiple sequence alignments and pairwise distance features from structural data to capture evolutionary and spatial constraints.

* Script: [get_conservation_and_distance_scores.py](https://github.com/chicanagram/pips-design-toolkit/blob/main/tools/get_conservation_and_distance_scores.py)
* Notebook: [get_conservation_and_distance_scores.ipynb](https://github.com/chicanagram/pips-design-toolkit/blob/main/notebooks/get_conservation_and_distance_scores.ipynb)

#### 2. Get Binding, Stability and Aggregation features

Extracts features related to binding, stability, and aggregation propensity to characterize protein variants across physicochemical properties.

Requirements: YASARA Structure installation (license needed); FoldX installation (license needed); Tango executable (license needed); Waltz (perl script incl.)

* Script: [get_binding_stability_agg_features.py](https://github.com/chicanagram/pips-design-toolkit/blob/main/tools/get_binding_stability_agg_features.py)
* Notebook: [get_binding_stability_agg_features.py.ipynb](https://github.com/chicanagram/pips-design-toolkit/blob/main/notebooks/get_binding_stability_agg_features.py.ipynb)

#### 3. Get PLM zero-shot scores and embedding features

Uses pretrained protein language models (PLMs) to generate zero-shot mutation likelihood scores and sequence embeddings for downstream analysis.

Requirements: fair-esm library installed (incl. in conda environment)

* Script: [get_pLM_zeroshot_scores_embeddings.py](https://github.com/chicanagram/pips-design-toolkit/blob/main/tools/get_pLM_zeroshot_scores_embeddings.py)
* Notebook: [get_pLM_zeroshot_scores_embeddings.ipynb](https://github.com/chicanagram/pips-design-toolkit/blob/main/notebooks/get_pLM_zeroshot_scores_embeddings.ipynb)

#### 4. Select mutants for first-round of protein engineering using zero-shot scores

Applies PLM zero-shot scores and other scores to prioritize and select candidate mutations likely to preserve or enhance protein function.

* Script: [select_zeroshot_mutants.py](https://github.com/chicanagram/pips-design-toolkit/blob/main/tools/select_zeroshot_mutants.py)
* Notebook: [select_zeroshot_mutants.ipynb](https://github.com/chicanagram/pips-design-toolkit/blob/main/notebooks/select_zeroshot_mutants.ipynb)

#### 5. Compile all features extracted into a dataset

Compiles raw feature scores from multiple sources into a unified dataset for further exploratory analysis and machine learning model evaluation.

* Script: [compile_analyse_dataset.py](https://github.com/chicanagram/pips-design-toolkit/blob/main/tools/compile_analyse_dataset.py)
* Notebook: [compile_analyse_dataset.ipynb](https://github.com/chicanagram/pips-design-toolkit/blob/main/notebooks/compile_analyse_dataset.ipynb)

#### 6. Train AutoML models on mutagenesis assay data

Trains and evaluates machine learning models with AutoGluon library for predicting protein properties, enabling automated model selection and hyperparameter tuning.

The full list of model features can be found in the `feature_names_multimodal` variable in: `./tools/ml_prediction/model_features.py`. 
The features utilized in the AutoML models include the Binding, Stability and Aggregation (Waltz and Tango) features. 

* Script: [evaluate_autogluon_classification.py](https://github.com/chicanagram/pips-design-toolkit/blob/main/tools/evaluate_autogluon_classification.py)
* Notebook: [evaluate_classification.ipynb](https://github.com/chicanagram/pips-design-toolkit/blob/main/notebooks/evaluate_classification.ipynb)

#### 7. Train traditional ML models on mutagenesis assay data

Implements and benchmarks traditional machine learning models (e.g., logistic regression, random forests, gradient boosting) using scikit-learn.

The full list of model features can be found in the `feature_names` variable combined via the `get_feature_combinations` function in: `./tools/ml_prediction/model_features.py`. 

Adjust which feature sets to run model evaluation on (or not) by commenting out specific featuresets in the `get_feature_combinations` function. 
The features utilized in the sklearn models include various combinations of:
```
- oheMT (one-hot encoding)
- esm2-33_LLRsum_entropy (log-likelihood ratio + PLM-derived entropy)
- esm2-33_seq_embeddings_MT (sequence-pooled embedding vector)
- esm2-33_mut_embeddings_MT (embedding vector corresponding to mutated residue)
```
* Script: [evaluate_sklearn_classification.py](https://github.com/chicanagram/pips-design-toolkit/blob/main/tools/evaluate_sklearn_classification.py)
* Notebook: [evaluate_classification.ipynb](https://github.com/chicanagram/pips-design-toolkit/blob/main/notebooks/evaluate_classification.ipynb)

## 4. Feature extraction software dependencies
While the multimodal activity prediction model can be trained using the dataset provided (see Section B) using the open-source Autogluon library, full extraction of the dataset features requires some other supporting software.
This includes 1) YASARA Structure, 2) FoldX 3) Tango and 4) Waltz.
  
#### YASARA Structure
* Obtain the license for YASARA Structure from: http://www.yasara.org/products.htm
* Download and unzip the Yasara folder, following the installation instructions provided. The final folder should contain the appropriate Yasara executable. 
* Note the location of the Yasara executable, and update it in the file **yasara.py**. i.e. change the line 
```
yasaradir = '/Applications/YASARA.app/Contents/yasara/'
```
#### FoldX
* Obtain the license for FoldX from: http://foldxsuite.crg.eu/
* Download the FoldX executable. Update its full path in the file **yasara.py**. i.e. change the line
```
foldx_abspath = yasaradir + '/foldx_2025/foldx_20251231_mac'
```
#### Tango and Waltz
* Obtain the licenses for Tango and Waltz from: https://switchlab.org/software/
* Download the respective executables and place them in the subfolder **feature_extraction > aggregation**. 
