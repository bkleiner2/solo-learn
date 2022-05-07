import os
import pickle
import numpy as np

def read_data(results_dir, model, logits_or_feats, num_files):
    data = []
    for i in range(1, num_files+1):
        with open(os.path.join(results_dir, f"simsiam-cifar100-linear-eval-{model}-subset{i}_{logits_or_feats}.pkl"), "rb") as pkl:
            file_data = pickle.load(pkl)
        data.append(file_data)
    return np.stack(data, axis=-1)

def compute_variance(data):
    return np.var(data, axis=-1).mean()

def main():
    results_dir = "../bash_files/linear/cifar/results" 
    for output in ["logits", "feats"]:
        for model in ["baseline", "momentum99"]:
            data = read_data(results_dir, model, output, 5)
            var = compute_variance(data)
            print (f"{model:10s} {output:6s} = {var:5f}")
        print ("---------------------------")
if __name__ == "__main__":
    main()
