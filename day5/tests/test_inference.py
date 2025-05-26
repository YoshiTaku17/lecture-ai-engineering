import time, joblib, numpy as np
from sklearn.metrics import accuracy_score

clf = joblib.load("model.pkl")
X = np.load("data/X_sample.npy")
y = np.load("data/y_sample.npy")

def test_accuracy():
    assert accuracy_score(y, clf.predict(X)) >= 0.9

def test_latency():
    t0 = time.perf_counter()
    clf.predict(X[:1])
    assert time.perf_counter() - t0 < 0.05
