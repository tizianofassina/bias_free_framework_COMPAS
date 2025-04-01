import numpy as np
import pandas as pd
from itertools import product
import cvxpy as cp



#Here we find the alternative optimization method based on the distribution p_{X, \hat(Y)}

df = pd.read_csv("compas-scores-cleaned-train.csv")

races = np.array(df["race"])
n = len(races)
recid = np.array(df["is_recid"])

races_dim = np.unique(df["race"])
recid_dim = np.unique(df["is_recid"])
age_dim = np.unique(df["age_cat"])
charge_dim = np.unique(df["c_charge_degree"])
priors_dim = np.unique(df["priors_count"])



x_dim = age_dim.shape[0]*charge_dim.shape[0]*priors_dim.shape[0]
d_dim = races_dim.shape[0]
y_dim = recid_dim.shape[0]



x_y_possible_values = np.array(list(product(age_dim, charge_dim, priors_dim, recid_dim )))


"""
for i in range(age_dim.shape[0]):
    for j in range(charge_dim.shape[0]):
        for k in range(priors_dim.shape[0]):
            for l in range(y_dim):
                for m in range(d_dim):
                    idx_1 += 1
                    for h in range(age_dim.shape[0]):
                        for b in range(charge_dim.shape[0]):
                            for c in range(priors_dim.shape[0]):
                                for y in range(y_dim):
                                idx_2+=1
                                
                                """
def direct_index_2(i, j, k, l, m):
    idx_1 = (i * (charge_dim.shape[0] * priors_dim.shape[0] * y_dim * d_dim) +
           j * (priors_dim.shape[0] * y_dim * d_dim) +
           k * (y_dim * d_dim) +
           l * d_dim +
           m)
    return idx_1


def inverse_index_2(idx_1):
    m = idx_1 % d_dim
    l = (idx_1 // d_dim) % y_dim
    k = (idx_1 // (y_dim * d_dim)) % priors_dim.shape[0]
    j = (idx_1 // (priors_dim.shape[0] * y_dim * d_dim)) % charge_dim.shape[0]
    i = (idx_1 // (charge_dim.shape[0] * priors_dim.shape[0] * y_dim * d_dim)) % age_dim.shape[0]

    return i, j, k, l, m

def direct_index_1(h, b, c, y):
    idx_2 = (h * (charge_dim.shape[0] * priors_dim.shape[0] * y_dim) +
             b * (priors_dim.shape[0] * y_dim) +
             c * y_dim +
             y)
    return idx_2


def inverse_index_1(idx_2):
    y = idx_2 % y_dim
    c = (idx_2 // y_dim) % priors_dim.shape[0]
    b = (idx_2 // (y_dim * priors_dim.shape[0])) % charge_dim.shape[0]
    h = (idx_2 // (y_dim * priors_dim.shape[0] * charge_dim.shape[0])) % age_dim.shape[0]

    return h, b, c, y

p_x_y = np.zeros(shape = x_dim*y_dim)
p_x_y_d = np.zeros(shape = x_dim*d_dim*y_dim)



p_y = np.zeros(shape = y_dim)
p_x_d = np.zeros(shape = x_dim*d_dim)
p_d = np.zeros(shape = d_dim)







for l in range(y_dim):
    bools_l = (df["is_recid"] == recid_dim[l])
    proba = np.sum(bools_l)/ n
    p_y[l] = proba

for l in range(d_dim):
    bools_l = (df["race"] == races[l])
    proba = np.sum(bools_l)/ n
    p_d[l] = proba


# Didn't find a better way of doing this without for loops
for i in range(age_dim.shape[0]):
    for j in range(charge_dim.shape[0]):
        for k in range(priors_dim.shape[0]):
            for l in range(y_dim):
                bools_i = (df["age_cat"] == age_dim[i])
                bools_j = (df["c_charge_degree"] == charge_dim[j])
                bools_k = (df["priors_count"] == priors_dim[k])
                bools_l = (df["is_recid"] == recid_dim[l])
                proba = np.sum((bools_i * bools_j * bools_k * bools_l)) / n

                p_x_y[direct_index_1(i,j,k,l)] = proba

for i in range(age_dim.shape[0]):
    for j in range(charge_dim.shape[0]):
        for k in range(priors_dim.shape[0]):
            for l in range(y_dim):
                for m in range(d_dim):
                    bools_i = (df["age_cat"] == age_dim[i])
                    bools_j = (df["c_charge_degree"] == charge_dim[j])
                    bools_k = (df["priors_count"] == priors_dim[k])
                    bools_m = (df["race"] == races_dim[m])
                    bools_l = (df["is_recid"] == recid_dim[l])
                    proba = np.sum((bools_i * bools_j * bools_k * bools_l * bools_m)) / n
                    p_x_y_d[direct_index_2(i,j,k,l,m)] = proba


p_x = np.zeros(x_dim)
index = 0
for i in range(age_dim.shape[0]):
    for j in range(charge_dim.shape[0]):
        for k in range(priors_dim.shape[0]):
                    bools_i = (df["age_cat"] == age_dim[i])
                    bools_j = (df["c_charge_degree"] == charge_dim[j])
                    bools_k = (df["priors_count"] == priors_dim[k])

                    proba = np.sum((bools_i * bools_j * bools_k )) / n
                    p_x[index] = proba
                    index += 1



# Didn't find a better way of doing this without for loops
for i in range(age_dim.shape[0]):
    for j in range(charge_dim.shape[0]):
        for k in range(priors_dim.shape[0]):
            for l in range(y_dim):
                for m in range(d_dim):
                    bools_i = (df["age_cat"] == age_dim[i])
                    bools_j = (df["c_charge_degree"] == charge_dim[j])
                    bools_k = (df["priors_count"] == priors_dim[k])
                    bools_m = (df["race"] == races_dim[m])
                    bools_l = (df["is_recid"] == recid_dim[l])
                    proba = np.sum((bools_i * bools_j * bools_k * bools_l * bools_m)) / n
                    p_x_y_d[direct_index_2(i,j,k,l,m)] = proba





matrix_norms = np.zeros(shape=(x_dim * y_dim, x_dim * y_dim * d_dim))
for i in range(age_dim.shape[0]):
    for j in range(charge_dim.shape[0]):
        for k in range(priors_dim.shape[0]):
            for l in range(y_dim):
                for m in range(d_dim):

                    for h in range(age_dim.shape[0]):
                        for b in range(charge_dim.shape[0]):
                            for c in range(priors_dim.shape[0]):
                                for y in range(y_dim):

                                    matrix_norms[direct_index_1(h,b,c,y), direct_index_2(i,j,k,l,m)] = (
                                            np.linalg.norm(np.array(
                                                [age_dim[h] - age_dim[i], charge_dim[j] - charge_dim[b],
                                                 priors_dim[k] - priors_dim[c], recid_dim[l] - recid_dim[y]])) ** 2
                                    )












if __name__ == "__main__":

    p_x_y_const = cp.Constant(p_x_y)
    p_x_y_d_const = cp.Constant(p_x_y_d)
    p_y_const = cp.Constant(p_y)
    p_d_const = cp.Constant(p_d)
    p_x_const = cp.Constant(p_x)
    matrix_norms_const = cp.Constant(matrix_norms)


    def Delta_alternative(p_cond):
        delta = 0
        for i in range(age_dim.shape[0]):
            for j in range(charge_dim.shape[0]):
                for k in range(priors_dim.shape[0]):
                    for y in range(y_dim):
                        index = (i * (charge_dim.shape[0] * priors_dim.shape[0] * y_dim) +
                                j * (priors_dim.shape[0] * y_dim) +
                                k * y_dim +
                                y)
                        delta+= cp.abs((p_y_hat_x(p_cond, i,j,k, y) - p_x_y_const[index]))
        return delta/2


    def delta(P_cond):
        return cp.sum(cp.multiply(matrix_norms_const, P_cond), axis=0)


    def J(proba):
        return cp.abs(proba / p_y_const - 1)


    def p_y_hat_d(p_cond, d, y):

        product = cp.multiply(p_cond, p_x_y_d_const[np.newaxis,:])
        p_y_hat_d = 0

        # For each (y, d) pair, sum over the first six dimensions
        # We need to select the appropriate indices based on your explanation
        for i in range(age_dim.shape[0]):
            for j in range(charge_dim.shape[0]):
                for k in range(priors_dim.shape[0]):
                    for l in range(y_dim):
                        for h in range(age_dim.shape[0]):
                            for b in range(charge_dim.shape[0]):
                                for c in range(priors_dim.shape[0]):
                                    # Apply the relevant summing operation

                                    # Update the final matrix with the sum over the indices
                                    p_y_hat_d  += product[direct_index_1(h,b,c,y), direct_index_2(i,j,k,l,d)]

        return  p_y_hat_d/p_d_const[d]




    def p_y_hat_x(p_cond, i,j,k, y):

        product = cp.multiply(p_cond, p_x_y_d_const[np.newaxis,:])
        result = 0
        for l in range(y_dim):
            for m in range(d_dim):
                for h in range(age_dim.shape[0]):
                    for b in range(charge_dim.shape[0]):
                        for c in range(priors_dim.shape[0]):
                                result+=product[direct_index_1(h, b, c, y), direct_index_2(i, j, k, l, d)]

        index = (
                i * (charge_dim.shape[0] * priors_dim.shape[0]) +
                j * priors_dim.shape[0] +
                k
                 )
        return result/p_x_const[index]




    c = 100
    eps = 1000

    P_cond = cp.Variable((x_dim * y_dim, x_dim * y_dim * d_dim))
    constraints = [P_cond >=0, P_cond<=1]

    for j in range(x_dim * y_dim * d_dim):
        constraints.append(cp.sum(P_cond[:, j]) == 1)

    for d in range(d_dim):
       for y in range(y_dim):
            constraints.append(p_y_hat_d(P_cond, d, y)<=eps)

    constraints.append(delta(P_cond) <=c)

    objective = cp.Minimize(Delta_alternative(P_cond))

    problem = cp.Problem(objective, constraints)
    problem.solve()
    P_cond_opt = np.array(P_cond.value)


    P_to_save = np.zeros((age_dim.shape[0],charge_dim.shape[0],priors_dim.shape[0], y_dim, age_dim.shape[0], charge_dim.shape[0], priors_dim.shape[0], y_dim, d_dim))

    idx_2 = 0
    for i in range(age_dim.shape[0]):
        for j in range(charge_dim.shape[0]):
            for k in range(priors_dim.shape[0]):
                for l in range(y_dim):
                    for m in range(d_dim):
                        idx_1 = 0
                        for h in range(age_dim.shape[0]):
                            for b in range(charge_dim.shape[0]):
                                for c in range(priors_dim.shape[0]):
                                    for y in range(y_dim):
                                        P_to_save[h,b,c,y,i,j,k,l,m] = P_cond_opt[idx_1,idx_2]
                                        idx_1 += 1

                        idx_2 += 1

    P_to_save = np.transpose(P_to_save, (0, 1, 2, 9, 4, 5, 6, 8, 7))
    np.save('P_cond_optimized_alternative.npy', P_to_save)

















