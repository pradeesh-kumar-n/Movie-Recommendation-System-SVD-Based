

import numpy as np
from scipy.sparse.linalg import svds
from config import SVD_N_FACTORS


class SVDModel:


    def __init__(self, n_factors=SVD_N_FACTORS):
        self.n_factors = n_factors
        self.U = None
        self.sigma = None
        self.Vt = None
        self.user_rating_mean = None

    def fit(self, user_item_matrix_filled):

        print("\n" + "=" * 70)
        print("Training SVD Model")
        print("=" * 70)
        R = user_item_matrix_filled.values

        # Calculate user means from observed ratings, excluding missing-value zeros.
        rated_counts = np.count_nonzero(R, axis=1)
        self.user_rating_mean = np.divide(
            R.sum(axis=1),
            rated_counts,
            out=np.zeros(R.shape[0], dtype=float),
            where=rated_counts > 0
        )

        # Keep missing entries at zero so they do not become artificial negatives.
        R_demeaned = np.where(
            R > 0,
            R - self.user_rating_mean.reshape(-1, 1),
            0
        )

        # Perform SVD
        self.U, sigma, self.Vt = svds(R_demeaned, k=self.n_factors)

        # Create a diagonal matrix from the singular values
        self.sigma = np.diag(sigma)

        print(f"U (Users x Features) Shape: {self.U.shape}")
        print(f"Sigma (Features) Shape    : {self.sigma.shape}")
        print(f"Vt (Features x Movies)    : {self.Vt.shape}")


        eigen_values = np.diag(self.sigma) ** 2
        total_variance = np.sum(eigen_values)
        explained_variance = np.cumsum(eigen_values) / total_variance
        print(f"Explained Variance by {self.n_factors} factors: {explained_variance[-1]:.2f}")

    def predict(self):

        # Reconstruct the matrix
        R_hat = np.dot(np.dot(self.U, self.sigma), self.Vt)
        # Add the user means back
        predictions = R_hat + self.user_rating_mean.reshape(-1, 1)
        # Clip ratings to be within the 1-5 scale
        predictions = np.clip(predictions, 1, 5)
        print(f"\nPredictions Matrix Shape: {predictions.shape}")
        return predictions