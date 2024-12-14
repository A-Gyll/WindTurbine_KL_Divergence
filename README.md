# Wind Turbine KL Divergence Attack

This repository contains the implementation of a KL divergence-based distribution inference attack on wind turbine (WT) data. The project is designed to evaluate the privacy vulnerabilities of federated learning frameworks in WT applications.

## Repository Structure

The code is organized into the following main sections:

### 1. `Data`
- **Purpose**: Stores the cleaned dataset and data splits for adversarial and victim models with varying target feature proportions.
- **Usage**: If you download the dataset used in [https://arxiv.org/pdf/2404.10320](https://arxiv.org/pdf/2404.10320), you can use the data cleaning script (`get_data.py`) to preprocess the data. Ensure that file paths are updated accordingly.

### 2. `EDA`
- **Purpose**: Contains code for initial Exploratory Data Analysis (EDA).
- **Details**: Includes scripts for feature selection and visualization of feature distributions, which informed decisions about target features for the attack.

### 3. `Tune_Hyperparams`
- **Purpose**: Includes scripts for building predictive models and tuning their hyperparameters.
- **Details**:
  - `check_model_accuracy.py` helps validate model performance.
  - `tune_hyper_parameters.ipynb` facilitates hyperparameter optimization to inform decisions for configuration files.

### 4. `Configs`
- **Purpose**: Stores configuration files for running KL divergence attacks with different setups.
- **Usage**: Modify these configuration files to experiment with various model architectures, data splits, and attack scenarios.

### 5. `Train_Models`
- **Purpose**: Contains support functions required to execute the KL divergence attack.
- **Details**:
  - Includes utility scripts for creating shadow models and victim models, as well as computing KL divergence between output distributions.

### 6. `run_kl_attack.py`
- **Purpose**: The main script for running the KL divergence attack.
- **Usage**: Execute the following command to run the attack:
  ```bash
  python run_kl_attack.py <config_path>
