

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score


def split_user_item_matrix(user_item_matrix, test_fraction=0.2, random_state=42):
    values = user_item_matrix.values
    train_matrix = user_item_matrix.copy()
    test_matrix = user_item_matrix.copy() * np.nan
    rng = np.random.default_rng(random_state)

    for user_index, user_ratings in enumerate(values):
        rated_movies = np.flatnonzero(user_ratings > 0)
        if len(rated_movies) < 2:
            continue

        test_count = min(
            len(rated_movies) - 1,
            max(1, int(round(len(rated_movies) * test_fraction)))
        )
        held_out = rng.choice(rated_movies, size=test_count, replace=False)
        train_matrix.iloc[user_index, held_out] = np.nan
        test_matrix.iloc[user_index, held_out] = values[user_index, held_out]

    return train_matrix, test_matrix


def evaluate_model(predictions, test_matrix, rating_threshold=4):

    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)

    actual = test_matrix.values

    mask = np.isfinite(actual) & (actual > 0)

    actual_ratings = actual[mask]
    predicted_ratings = predictions[mask]

    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    mae = np.mean(np.abs(actual_ratings - predicted_ratings))
    actual_positive = actual_ratings >= rating_threshold
    predicted_positive = predicted_ratings >= rating_threshold
    accuracy = accuracy_score(actual_positive, predicted_positive)
    precision = precision_score(actual_positive, predicted_positive, zero_division=0)
    recall = recall_score(actual_positive, predicted_positive, zero_division=0)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"Accuracy (rating >= {rating_threshold}): {accuracy:.4f}")
    print(f"Precision (rating >= {rating_threshold}): {precision:.4f}")
    print(f"Recall (rating >= {rating_threshold}): {recall:.4f}")
    print("=" * 70 + "\n")

    return rmse, mae, accuracy, precision, recall