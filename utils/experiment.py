
import numpy as np
import math
import torch

def test_performance(anomalies, scores, percentage):

    # Order anomalies (binary vector) by the anomaly score in descending order
    _, ordering = torch.sort(scores, descending=True)
    ordered_anomalies = anomalies[ordering.type(torch.LongTensor)]

    # Number of observations to include in top
    n = anomalies.shape[0]
    n_top = math.ceil(n * percentage / 100)

    return torch.sum(ordered_anomalies[:n_top]).item() / n

if __name__=='__main__':
    n = 1000000
    anomalies = torch.from_numpy(np.random.choice([0,1],size=n))
    scores = torch.from_numpy(np.random.uniform(0,5,size=n))
    percentages = [1,5,10,20]

    # Order anomalies (binary vector) by the anomaly score in descending order
    _, ordering = torch.sort(scores, descending=True)
    ordered_anomalies = anomalies[ordering.type(torch.LongTensor)]

    results = [test_performance(anomalies, scores, x) for x in percentages]

    np.testing.assert_allclose(results, [0.005, 0.025, 0.05, 0.1], rtol=0.3)
