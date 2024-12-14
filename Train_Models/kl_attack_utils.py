import numpy as np

def kl_divergence(p, q, epsilon=1e-12):
    """
    Compute KL divergence between two probability distributions.
    """
    p = np.clip(p, epsilon, 1 - epsilon)
    q = np.clip(q, epsilon, 1 - epsilon)
    return np.sum(p * np.log(p / q), axis=-1)  # Sum over the last axis (classes)

def evaluate_kl_divergence(victim_preds, shadow_preds_1, shadow_preds_2, closer_ratio):
    # flatten victim predictions ... only one model
    victim_preds = victim_preds.squeeze(0)

    # compute KL divergence for shadow model predictions vs victim predictions
    kl_1 = kl_divergence(
        victim_preds[:, np.newaxis, :],  
        shadow_preds_1.transpose(1, 0, 2)
    ) 

    kl_2 = kl_divergence(
        victim_preds[:, np.newaxis, :], 
        shadow_preds_2.transpose(1, 0, 2)  
    )  

    # avg KL results across models for each shadow set
    avg_kl_1 = np.mean(kl_1, axis=1)
    avg_kl_2 = np.mean(kl_2, axis=1)

    # lower avg kl score means closer to dist
    if closer_ratio == 0:
        results = (avg_kl_1 < avg_kl_2).astype(int).tolist()
    elif closer_ratio == 1:
        results = (avg_kl_2 < avg_kl_1).astype(int).tolist()
    else:
        raise ValueError("closer_ratio must be 0 or 1")

    return results


