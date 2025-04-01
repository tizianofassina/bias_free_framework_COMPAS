import numpy as np
import pandas as pd
from optimization_classical import races_dim, recid_dim,age_dim,charge_dim,priors_dim



# Uploading the transforming distribution
p_cond = np.load("P_cond_optimized_classical.npy")


def preprocess_compas(df, p_cond, filename):
    rows = []
    for _, row in df.iterrows():
        age_index = np.where(age_dim == row["age_cat"])[0][0]
        charge_index = np.where(charge_dim == row["c_charge_degree"])[0][0]
        priors_index = np.where(priors_dim == row["priors_count"])[0][0]
        recid_index = np.where(recid_dim == row["is_recid"])[0][0]
        race_index = np.where(races_dim == row["race"])[0][0]

        weights = p_cond[:, :, :, :, age_index, charge_index, priors_index, race_index, recid_index]
        random_idx = np.random.choice(weights.size, p=weights.flatten())

        i, j, k, l = np.unravel_index(random_idx, weights.shape)

        new_row = {
            "age_cat": age_dim[i],
            "c_charge_degree": charge_dim[j],
            "priors_count": priors_dim[k],
            "is_recid": recid_dim[l]
        }
        rows.append(new_row)

    processed_df = pd.DataFrame(rows)
    processed_df.to_csv(filename, index=False)


if __name__ == "__main__":

    df_train = pd.read_csv("compas-scores-cleaned-train.csv")
    preprocess_compas(df_train, p_cond, "pre-processed-classical-compas-scores-cleaned-trained.csv")

    df_test = pd.read_csv("compas-scores-cleaned-test.csv")
    preprocess_compas(df_test, p_cond, "pre-processed-classical-compas-scores-cleaned-test.csv")