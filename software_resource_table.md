# Software Resource Table

This table was compiled by reviewing the project setup and workflow documentation in [README.md](/Users/charmainechia/Documents/projects/pips-design-toolkit/README.md:14), plus the pinned dependencies in [requirements.txt](/Users/charmainechia/Documents/projects/pips-design-toolkit/requirements.txt:1) and [environment.yml](/Users/charmainechia/Documents/projects/pips-design-toolkit/environment.yml:1).

It is intentionally a "key resources" list rather than an exhaustive dump of every transitive dependency.

## External software and web tools

| Software | Why it is used in this repo | Suggested citation | Official link |
| --- | --- | --- | --- |
| Python 3.11 | Base runtime for all scripts and notebooks; explicitly pinned in the environment setup. | Python Software Foundation. *Python Language Reference, version 3.11*. | [python.org](https://www.python.org/) |
| Jupyter Notebook | Notebook-based execution path documented in the README. | Kluyver T, Ragan-Kelley B, Perez F, et al. *Jupyter Notebooks - a publishing format for reproducible computational workflows* (2016). | [jupyter.org](https://jupyter.org/) |
| YASARA Structure | Required for binding and stability feature extraction workflows. | Krieger E, Vriend G. *YASARA View - molecular graphics for all devices from smartphones to workstations*. Bioinformatics (2014). | [YASARA Products](https://www.yasara.org/products.htm) |
| FoldX | Required for protein stability / mutational free-energy calculations. | Schymkowitz J, Borg J, Stricher F, Nys R, Rousseau F, Serrano L. *The FoldX web server: an online force field*. Nucleic Acids Research (2005). | [foldxsuite.crg.eu](https://foldxsuite.crg.eu/) |
| SIFT | Used through the SIFT webserver for conservation / mutational tolerance scoring. | Ng PC, Henikoff S. *SIFT: Predicting amino acid changes that affect protein function*. Nucleic Acids Research (2003). | [SIFT webserver](https://sift.bii.a-star.edu.sg/) |
| TANGO | Required for aggregation-prone region prediction in the feature extraction workflow. | Fernandez-Escamilla AM, Rousseau F, Schymkowitz J, Serrano L. *Prediction of sequence-dependent and mutational effects on the aggregation of peptides and proteins*. Nature Biotechnology (2004). | [Switch Lab software](https://switchlab.org/software/) |
| WALTZ | Required for amyloidogenic region prediction in the feature extraction workflow. | Maurer-Stroh S, Debulpaep M, Kuemmerer N, et al. *Exploring the sequence determinants of amyloid structure using position-specific scoring matrices*. Nature Methods (2010). | [Switch Lab software](https://switchlab.org/software/) |
| AutoGluon | Main AutoML framework used for the multimodal classification pipeline. | Erickson N, Mueller J, Shirkov A, et al. *AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data* (2020). | [AutoGluon docs](https://auto.gluon.ai/stable/index.html) |
| ESM / ESM-2 | Protein language model stack used for zero-shot scores and embedding features via `fair-esm`. | Lin Z, Akin H, Rao R, et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science (2023). | [facebookresearch/esm](https://github.com/facebookresearch/esm) |

## Key Python modules

| Python module | Why it is used in this repo | Suggested citation | Official link |
| --- | --- | --- | --- |
| pandas | Core table loading, merging, filtering, and CSV I/O across the data workflows. | McKinney W. *Data Structures for Statistical Computing in Python*. Proc. SciPy (2010). | [pandas citing page](https://pandas.pydata.org/about/citing.html) |
| NumPy | Core numerical array operations throughout the feature extraction and ML code. | Harris CR, Millman KJ, van der Walt SJ, et al. *Array programming with NumPy*. Nature (2020). | [numpy.org](https://numpy.org/) |
| SciPy | Scientific computing utilities used by the ML and analysis stack. | Virtanen P, Gommers R, Oliphant TE, et al. *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*. Nature Methods (2020). | [scipy.org](https://scipy.org/) |
| scikit-learn | Main traditional ML framework and shared metrics / CV utilities. | Pedregosa F, Varoquaux G, Gramfort A, et al. *Scikit-learn: Machine Learning in Python*. JMLR (2011). | [scikit-learn citing page](https://scikit-learn.org/stable/about.html#citing-scikit-learn) |
| matplotlib | Plotting and visualization in notebooks and SHAP output scripts. | Hunter JD. *Matplotlib: A 2D Graphics Environment*. Computing in Science and Engineering (2007). | [matplotlib citing page](https://matplotlib.org/stable/users/project/citing.html) |
| Biopython | Sequence / structure handling utilities in the bioinformatics workflows. | Cock PJA, Antao T, Chang JT, et al. *Biopython: freely available Python tools for computational molecular biology and bioinformatics*. Bioinformatics (2009). | [Biopython documentation](https://biopython.org/wiki/Documentation) |
| XGBoost | Optional model family used in the AutoGluon and sklearn-style workflows. | Chen T, Guestrin C. *XGBoost: A Scalable Tree Boosting System*. KDD (2016). | [XGBoost docs](https://xgboost.readthedocs.io/) |
| CatBoost | Gradient boosting model family listed in the environment dependencies. | Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. *CatBoost: unbiased boosting with categorical features* (2018). | [catboost.ai](https://catboost.ai/) |
| SHAP | Model explanation workflow used by the SHAP analysis scripts. | Lundberg SM, Lee SI. *A Unified Approach to Interpreting Model Predictions*. NeurIPS (2017). | [SHAP docs](https://shap.readthedocs.io/) |
| fair-esm | Python package wrapper used to load Meta ESM protein language models. | Lin Z, Akin H, Rao R, et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science (2023). | [PyPI / GitHub project](https://github.com/facebookresearch/esm) |

## Repo evidence summary

- Environment management and Python version: [README.md](/Users/charmainechia/Documents/projects/pips-design-toolkit/README.md:16), [README.md](/Users/charmainechia/Documents/projects/pips-design-toolkit/README.md:32), [environment.yml](/Users/charmainechia/Documents/projects/pips-design-toolkit/environment.yml:5)
- Notebook workflow: [README.md](/Users/charmainechia/Documents/projects/pips-design-toolkit/README.md:68)
- Feature-extraction dependencies: [README.md](/Users/charmainechia/Documents/projects/pips-design-toolkit/README.md:84), [README.md](/Users/charmainechia/Documents/projects/pips-design-toolkit/README.md:143)
- AutoGluon workflow: [README.md](/Users/charmainechia/Documents/projects/pips-design-toolkit/README.md:116)
- scikit-learn workflow: [README.md](/Users/charmainechia/Documents/projects/pips-design-toolkit/README.md:126)
- Pinned Python dependencies: [requirements.txt](/Users/charmainechia/Documents/projects/pips-design-toolkit/requirements.txt:1), [environment.yml](/Users/charmainechia/Documents/projects/pips-design-toolkit/environment.yml:7)

## Notes

- I treated this as a key-resource table for the main scientific workflows, not as a full lockfile-style list of every low-level dependency.
- For manuscript use, you may want to replace generic software citations with version-specific citations where the package maintainers explicitly recommend them, for example pandas on Zenodo.
