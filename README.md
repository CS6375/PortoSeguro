# PortoSeguro

## Project structure

    PortoSeguro
    ├── README.md                       # Current file.
    ├── doc
    │   └── report.pdf                  # Final report.
    ├── load_data.py                    # Script to load data.
    ├── log                             # Training log of each model.
    │   ├── gradient_boost.log
    │   ├── logistic_regression.log
    │   ├── neural_network.log
    │   └── random_forest.log
    ├── notebook
    │   ├── Plot-Hist-Corr.ipynb        # Notebook for all plots in report.
    │   └── dimension_reduction.ipynb   # Attempted dimension reduction.
    ├── train_gb.py                     # Model of Gradient Boosting.
    ├── train_lr.py                     # Model of Logistic Regression.
    ├── train_nn.py                     # Model of Neural Network.
    └── train_rf.py                     # Model of Random Forests.

## Environment Dependencies

The python scripts run on Python 3.6.3.
The following libraries are necessary to run scripts or notebooks.

-   matplotlib (2.1.0)
-   numpy (1.13.3)
-   pandas (0.20.3)
-   scikit-image (0.13.0)
-   scikit-learn (0.19.1)
-   scipy (0.19.1)
-   seaborn (0.8)
-   tqdm (4.19.4)

## Execution

The data file should be in `./data` with filenames `test.csv` and `train.csv`.
Directly execute the script as follow:

    python3 train_gb.py
    python3 train_lr.py
    python3 train_nn.py
    python3 train_rf.py
