import pandas as pd
import os
import joblib
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from Train_Models.classification_models import ClassificationModels

class WindTurbineDataSplitter:
    def __init__(self, drop_sensitive_cols=False):
        self.drop_sensitive_cols = drop_sensitive_cols
        self.columns = [
            "ambient_temperature", "wind_relative_direction",
            "wind_speed", "total_active_power", 
            "generator_rpm", "rotor_rpm", "gearbox_temp_bin", "pitch_angle_bin"
        ]

    def load_data(self, df, test_ratio=0.4, random_state=42):
        """
        load and split the dataframe into train and test sets based on pitch_angle_bin and gearbox_temp_bin
        """
        if not all(col in df.columns for col in self.columns):
            raise ValueError("Input dataframe does not contain all required columns.")

        df = df[self.columns].copy()

        def stratified_split(data, rs):
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_ratio,
                random_state=rs
            )
            splitter = sss.split(data, data[["pitch_angle_bin", "gearbox_temp_bin"]])
            split_1, split_2 = next(splitter)
            return data.iloc[split_1], data.iloc[split_2]

        self.train_df, self.test_df = stratified_split(df, random_state)

        if self.drop_sensitive_cols:
            self.train_df = self.train_df.drop(columns=['pitch_angle_bin', 'gearbox_temp_bin'])
            self.test_df = self.test_df.drop(columns=['pitch_angle_bin', 'gearbox_temp_bin'])

        return self.train_df, self.test_df

    def sample_with_ratio_multiple_splits(self, df, feature, ratio, total_samples, num_splits, random_seed=42):
        """
        sample multiple smaller datasets with a specified ratio of a given feature (non-overlapping sets)
        """
        if feature not in df.columns: raise ValueError(f"Feature '{feature}' not found in the dataframe.")

        unique_values = df[feature].unique()
        if len(unique_values) != 2: raise ValueError("Feature must be binary for ratio sampling.")

        df_1 = df[df[feature] == unique_values[0]]
        df_2 = df[df[feature] == unique_values[1]]

        samples_1 = int(total_samples * ratio)
        samples_2 = total_samples - samples_1

        if samples_1 * num_splits > len(df_1):
            raise ValueError(f"Not enough samples in group '{unique_values[0]}' to create {num_splits} splits.")
        if samples_2 * num_splits > len(df_2):
            raise ValueError(f"Not enough samples in group '{unique_values[1]}' to create {num_splits} splits.")

        splits = []
        for i in range(num_splits):
            df_1_sampled = df_1.sample(n=samples_1, random_state=random_seed + i)
            df_2_sampled = df_2.sample(n=samples_2, random_state=random_seed + i)

            sampled_df = pd.concat([df_1_sampled, df_2_sampled]).sample(frac=1, random_state=random_seed + i).reset_index(drop=True)
            splits.append(sampled_df)

            df_1 = df_1.drop(df_1_sampled.index)
            df_2 = df_2.drop(df_2_sampled.index)

        updated_df = pd.concat([df_1, df_2]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
        return splits, updated_df

def create_shadow_models_data_splits(train_df, splitter, split, target_property, num_splits=5, total_samples_per_split=40000, ratios=None):
    """
    Create shadow model data splits for different property ratios
    """
    if split == "victim":
        output_directory = f"/home/ujx4ab/ondemand/WindTurbine_KL_Divergence/Data/{split}_splits/{target_property}"
        os.makedirs(output_directory, exist_ok=True)

        single_split_filename = os.path.join(output_directory, f"{ratios[0]}.csv")
        if os.path.exists(single_split_filename):
            print(f"File already exists for 'victim' split at {single_split_filename}. Skipping generation.")
            return

        train_df.to_csv(single_split_filename, index=False)
        print(f"Created single model data for 'victim' split at {single_split_filename}")
        return

    if ratios is None:
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    output_directory = f"/home/ujx4ab/ondemand/WindTurbine_KL_Divergence/Data/{split}_splits/{target_property}"
    os.makedirs(output_directory, exist_ok=True)

    for ratio in tqdm(ratios, desc="Processing Ratios", unit="ratio"):
        ratio_directory = os.path.join(output_directory, f"{ratio}")
        os.makedirs(ratio_directory, exist_ok=True)

        # Check if all split files exist for this ratio
        all_files_exist = all(
            os.path.exists(os.path.join(ratio_directory, f"{i + 1}.csv"))
            for i in range(num_splits)
        )
        if all_files_exist:
            print(f"All files already exist for ratio {ratio}. Skipping generation.")
            continue

        updated_train_df = train_df.copy()
        current_total_samples = total_samples_per_split

        while current_total_samples > 0:
            try:
                splits, updated_train_df = splitter.sample_with_ratio_multiple_splits(
                    updated_train_df, feature=target_property, ratio=ratio,
                    total_samples=current_total_samples, num_splits=num_splits
                )

                for i, split_df in enumerate(splits):
                    split_filename = os.path.join(ratio_directory, f"{i + 1}.csv")
                    split_df.to_csv(split_filename, index=False)
                break

            except ValueError as e:
                print(f"Error with ratio {ratio} and {current_total_samples} samples: {e}")
                current_total_samples -= 2500

        if current_total_samples <= 0:
            print(f"Failed to create splits for ratio {ratio} even after reducing sample size.")

def train_shadow_models(property_ratio, model, model_name, target_property, pred_property, split):
    """
    Train shadow models for the given property ratio.
    """
    data_splits_directory = f"/home/ujx4ab/ondemand/WindTurbine_KL_Divergence/Data/{split}_splits/{target_property}/"
    attack_directory = f"/home/ujx4ab/ondemand/WindTurbine_KL_Divergence/Train_Models/shadow_models/{model_name}/{split}/{target_property}/{property_ratio}"
    os.makedirs(attack_directory, exist_ok=True)

    if split == "victim":
        # Handle "victim" case separately
        victim_file_path = os.path.join(data_splits_directory, f"{property_ratio}.csv")
        if not os.path.exists(victim_file_path):
            raise FileNotFoundError(f"Victim data not found at {victim_file_path}.")

        data = pd.read_csv(victim_file_path)

        X = data.drop(columns=[pred_property])
        y = data[pred_property]

        model_file_name = "victim_model.joblib"
        model_path = os.path.join(attack_directory, model_file_name)

        # Skip training if model already exists
        if os.path.exists(model_path):
            print(f"Victim model already exists at {model_path}. Skipping training.")
            return

        model.fit(X, y)
        joblib.dump(model, model_path)

        print(f"Trained and saved victim model at {model_path}.")
        return

    ratio_directory = os.path.join(data_splits_directory, f"{property_ratio}")

    if not os.path.exists(ratio_directory):
        raise FileNotFoundError(f"Data splits for ratio {property_ratio} not found at {ratio_directory}.")

    files = [file_name for file_name in os.listdir(ratio_directory) if file_name.endswith(".csv")]
    with tqdm(total=len(files), desc=f"Building {model_name} models", unit="file") as pbar:
        for file_name in files:
            file_path = os.path.join(ratio_directory, file_name)
            model_file_name = file_name.replace(".csv", ".joblib")
            model_path = os.path.join(attack_directory, model_file_name)

            # Skip training if model already exists
            if os.path.exists(model_path):
                pbar.write(f"Model for {file_name} already exists at {model_path}. Skipping.")
                pbar.update(1)
                continue
                
            data = pd.read_csv(file_path)

            X = data.drop(columns=[pred_property])
            y = data[pred_property]

            model.fit(X, y)
            joblib.dump(model, model_path)

            pbar.set_postfix(file=file_name)
            pbar.update(1)
            pbar.write(f"Trained and saved model for {file_name} at {model_path}.")

def get_model(model_name, **kwargs):
    classification_models = ClassificationModels()
    
    model_mapping = {
        "random forest classifier": classification_models.RandomForest,
        "logistic regression classifier": classification_models.LRClassifier,
        "k neighbors classifier": classification_models.KNeighborsClassifier,
    }

    model_constructor = model_mapping.get(model_name.lower())
    if model_constructor is None:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(model_mapping.keys())}")

    return model_constructor(**kwargs)
