import os
import joblib
import numpy as np
import pandas as pd
import yaml
import sys
import csv
from itertools import combinations
from Train_Models.utils import WindTurbineDataSplitter, create_shadow_models_data_splits, train_shadow_models, get_model
from Train_Models.kl_attack_utils import evaluate_kl_divergence
from sklearn.metrics import accuracy_score

def infer_distribution_accuracy(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract configuration parameters
    shadow_model_configs = config['shadow_models']  # shadow model configurations
    target_model_config = config.get('target_model')  # target model configuration

    # use the first shadow model as the target if target model not given 
    if not target_model_config:
        print("No target model specified. Using the first shadow model configuration as the target model.")
        target_model_config = shadow_model_configs[0]

    target_model_name = target_model_config['name']
    target_model_params = target_model_config['params']
    target_property = config['target_property']
    pred_property = config['pred_property']
    shadow_model_ratios = config['shadow_model_ratios']
    data_path = config['data_path']
    test_ratio = config.get('test_ratio', 0.2)

    filtered_df = pd.read_csv(data_path)
    filtered_df.drop('timestamp', axis=1, inplace=True)

    initial_ratios = {col: (filtered_df[col].value_counts(normalize=True)).to_dict() for col in [target_property, pred_property]}
    true_ratio = round(initial_ratios[target_property].get(1.0, 0), 2)
    assert true_ratio != 0, "True ratio for the target property could not be calculated."

    # train and test sets
    splitter = WindTurbineDataSplitter()
    train_df, test_df = splitter.load_data(filtered_df, test_ratio=test_ratio)

    # train shadow model based on config
    for shadow_model_config in shadow_model_configs:
        shadow_model_name = shadow_model_config['name']
        shadow_model_params = shadow_model_config['params']
        
        for ratio in shadow_model_ratios:
            create_shadow_models_data_splits(
                train_df=train_df,
                splitter=splitter,
                split="adversarial",
                target_property=target_property,
                num_splits=5,
                total_samples_per_split=40000,
                ratios=[ratio]
            )
            
            train_shadow_models(
                ratio,
                get_model(shadow_model_name, **shadow_model_params),
                split="adversarial",
                model_name=shadow_model_name,
                pred_property=pred_property,
                target_property=target_property
            )

    # train victim model based on config
    create_shadow_models_data_splits(
        train_df=train_df,
        splitter=splitter,
        split="victim",
        target_property=target_property,
        num_splits=5,
        total_samples_per_split=40000,
        ratios=[true_ratio]
    )

    train_shadow_models(
        true_ratio,
        get_model(target_model_name, **target_model_params),
        split="victim",
        model_name=target_model_name,
        pred_property=pred_property,
        target_property=target_property
    )

    def load_models_from_directory(path):
        models = []
        for filename in os.listdir(path):
            if filename.endswith(".joblib"):
                model_path = os.path.join(path, filename)
                models.append(joblib.load(model_path))
        return models

    def generate_predictions(models, dataset):
        predictions = []
        for model in models:
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(dataset)
            else:
                preds = model.predict(dataset).reshape(-1, 1)
            predictions.append(preds)
        return np.array(predictions)

    victim_model_path = f"/home/ujx4ab/ondemand/WindTurbine_KL_Divergence/Train_Models/shadow_models/{target_model_name}/victim/{target_property}/{true_ratio}/"
    victim_models = load_models_from_directory(victim_model_path)
    test_x = test_df.drop(pred_property, axis=1)
    victim_predictions = generate_predictions(victim_models, test_x)

    accuracy_results = []

    for shadow_model_config in shadow_model_configs:
        shadow_model_name = shadow_model_config['name']
        
        for ratio_pair in combinations(shadow_model_ratios, 2):
            shadow_model_paths_1 = f"/home/ujx4ab/ondemand/WindTurbine_KL_Divergence/Train_Models/shadow_models/{shadow_model_name}/adversarial/{target_property}/{ratio_pair[0]}"
            shadow_model_paths_2 = f"/home/ujx4ab/ondemand/WindTurbine_KL_Divergence/Train_Models/shadow_models/{shadow_model_name}/adversarial/{target_property}/{ratio_pair[1]}"

            shadow_models_1 = load_models_from_directory(shadow_model_paths_1)
            shadow_models_2 = load_models_from_directory(shadow_model_paths_2)

            shadow_predictions_set_1 = generate_predictions(shadow_models_1, test_x)
            shadow_predictions_set_2 = generate_predictions(shadow_models_2, test_x)

            closer_to_true_ratio = 0 if abs(ratio_pair[0] - true_ratio) < abs(ratio_pair[1] - true_ratio) else 1
            results = evaluate_kl_divergence(victim_predictions, shadow_predictions_set_1, shadow_predictions_set_2, closer_to_true_ratio)

            correct_guesses = results.count(1)
            accuracy = correct_guesses / len(results)

            accuracy_results.append({
                "shadow_model": shadow_model_name,
                "ratio_1": ratio_pair[0],
                "ratio_2": ratio_pair[1],
                "accuracy": accuracy
            })

    output_file = "accuracy_results.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["shadow_model", "ratio_1", "ratio_2", "accuracy"])
        writer.writeheader()
        writer.writerows(accuracy_results)

    print(f"Results saved to {output_file}")

    return accuracy_results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_kl_attack.py <path_to_config>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    accuracy = infer_distribution_accuracy(config_file_path)
    print("Accuracy of the inference attack:", accuracy)