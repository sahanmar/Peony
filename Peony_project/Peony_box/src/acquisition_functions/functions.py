import numpy as np


def random_sampling(instaces: np.ndarray, len_rand_samples: int) -> np.ndarray:
    instaces = np.mean(instaces, axis=0)
    return np.random.randint(instaces.shape[0], size=len_rand_samples).astype(int)
