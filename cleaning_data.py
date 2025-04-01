import numpy as np
import pandas as pd
import math



#This files just cleans data like in the experiment of the article


df = pd.read_csv("compas-scores.csv")

df = df[["age_cat", "race", "c_charge_degree", "is_recid", "priors_count"]]

df = df[(df["race"] == "Caucasian") | (df["race"] == "African-American")]

df = df[df["priors_count"] != -1]

df = df[df["is_recid"] != -1]

df = df[df["c_charge_degree"] != "O"]


race_mapping = {
    "Caucasian": 0,
    "African-American": 1
}
charge_mapping = {
    "M": 10,
    "F": 20,
}

age_mapping = {
    "Less than 25": 0,
    "25 - 45": 1,
    "Greater than 45": 2
}

df["age_cat"] = df["age_cat"].map(age_mapping)
df["c_charge_degree"] = df["c_charge_degree"].map(charge_mapping)
df["race"] = df["race"].map(race_mapping)


length = df.shape[0]
print("Length of the dataset", length)
train_index = np.random.choice(df.index, size = math.ceil(length*0.75), replace=False)
df_train = df.loc[train_index]
length = df_train.shape[0]
print("Length of the train dataset", length)

df_test = df.drop(train_index)
length = df_test.shape[0]
print("Length of the test dataset", length)


df.to_csv("compas-scores-cleaned.csv", index = False)
df_train.to_csv("compas-scores-cleaned-train.csv", index = False)
df_test.to_csv("compas-scores-cleaned-test.csv", index = False)






