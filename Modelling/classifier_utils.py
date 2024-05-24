import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def divide_dataset(df, target):
    """
    Splits the dataset into training and testing parts.

    Args:
        df: Dataset to split.
        target: Name of the target feature.

    Returns:
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
    """
    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    return X_train, y_train, X_test, y_test

def get_clf_metrics(model, X_train, y_train, X_test, y_test, clf_is_from_skl=True, random_state=None, **clf_params):
    """
    Calculate classifier performance metrics.

    Args:
        model: Classifier model.
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        clf_is_from_skl: Flag indicating whether the classifier is from scikit-learn.
        random_state: Random state for reproducibility (optional).
        **clf_params: Additional classifier parameters.

    Returns:
        rmse: Root Mean Squared Error.
        mae: Mean Absolute Error.
        r_squared: R-squared.
    """
    clf = model(**clf_params) if clf_is_from_skl else model(**clf_params)
    if random_state is not None:
        clf.set_params(random_state=random_state)
    clf.fit(X_train, y_train)

    # Calculate RMSE, MAE, and R-squared
    y_pred = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    return rmse, mae, r_squared

def get_clf_metric_for_val_curves(model, param, param_grid, X_train, y_train, X_test, y_test, clf_is_from_skl=True, **clf_params):
    """
    Calculates classifier performance metric values for validation curves.

    Args:
        model: Classifier model.
        param: Name of the selected model parameter.
        param_grid: Values of the selected model parameter.
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        clf_is_from_skl: Flag indicating whether the classifier is from scikit-learn.
        **clf_params: Additional classifier parameters.

    Returns:
        train_rmse: List of RMSE values on the training dataset.
        train_mae: List of MAE values on the training dataset.
        train_r_squared: List of R-squared values on the training dataset.
        test_rmse: List of RMSE values on the testing dataset.
        test_mae: List of MAE values on the testing dataset.
        test_r_squared: List of R-squared values on the testing dataset.
        feature_importances: List of feature importances.
    """
    train_rmse, train_mae, train_r_squared = [], [], []
    test_rmse, test_mae, test_r_squared = [], [], []
    feature_importances = []

    for param_value in param_grid:
        # Initialize the classifier with current parameter value
        clf = model(random_state=0, **{param: param_value}, **clf_params) if clf_is_from_skl else model(**{param: param_value}, **clf_params)
        
        # Train the classifier
        clf.fit(X_train, y_train)

        # Calculate the appropriate metrics on training and testing datasets
        train_rmse_value, train_mae_value, train_r_squared_value, _, train_feature_importances = get_clf_metrics(clf, X_train, y_train, X_train, y_train)
        test_rmse_value, test_mae_value, test_r_squared_value, _, test_feature_importances = get_clf_metrics(clf, X_train, y_train, X_test, y_test)

        # Append the metrics to the lists
        train_rmse.append(train_rmse_value)
        train_mae.append(train_mae_value)
        train_r_squared.append(train_r_squared_value)
        test_rmse.append(test_rmse_value)
        test_mae.append(test_mae_value)
        test_r_squared.append(test_r_squared_value)
        feature_importances.append([train_feature_importances, test_feature_importances])

    return train_rmse, train_mae, train_r_squared, test_rmse, test_mae, test_r_squared, feature_importances

def plot_val_curves(param, param_grid, train_metric, test_metric, metric_name):
    """
    Plots validation curves.

    Args:
        param: Name of the selected model parameter.
        param_grid: Values of the selected model parameter.
        train_metric: List of metric values on the training dataset.
        test_metric: List of metric values on the testing dataset.
        metric_name: Name of the metric to be plotted (e.g., "RMSE", "MAE", "R-squared").

    Returns:
        None
    """
    plt.figure(figsize=(8, 4))
    plt.plot(param_grid, train_metric, alpha=0.5, color="blue", label="train")
    plt.plot(param_grid, test_metric, alpha=0.5, color="red", label="test")
    plt.legend(loc="best")
    plt.xlabel(param)
    plt.ylabel(metric_name)
    plt.show()
