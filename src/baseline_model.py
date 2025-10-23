from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit

def train_baseline_model(X_train, y_train, model_type='lr', **kwargs):
    """
    Train a baseline model (Logistic Regression or Random Forest) with optional hyperparameters.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    model_type : str
        'lr' or 'rf'
    **kwargs :
        Additional model hyperparameters (e.g., C=0.1, n_estimators=300)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if model_type == 'lr':
        model = LogisticRegression(max_iter=1000, **kwargs)
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42, **kwargs)
    else:
        raise ValueError("model_type must be 'lr' or 'rf'")

    model.fit(X_train_scaled, y_train)
    return model, scaler


def gridsearch_model(X_train, y_train, model_type='lr', param_grid=None, cv=None):
    """
    Performs grid search with cross-validation for Logistic Regression or Random Forest.
    Automatically scales data and returns the best fitted model.
    cv needs to be predi
    """
    if param_grid is None:
        param_grid = {}
    
    assert type(cv) is PredefinedSplit, "Use predefined split for CV!"

    # Define pipeline
    if model_type == 'lr':
        pipe = Pipeline([
            ('scaler', StandardScaler()), # need to put scaler inside pipeline to avoid data leakage from globally computed standard scalar
            ('model', LogisticRegression(max_iter=1000))
        ])
    elif model_type == 'rf':
        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),  # optional, RF doesn’t need scaling
            ('model', RandomForestClassifier(random_state=42))
        ])
    else:
        raise ValueError("model_type must be 'lr' or 'rf'")

    # Adjust param grid for pipeline naming
    param_grid = {f"model__{k}": v for k, v in param_grid.items()}

    # Grid search
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv, # predefined split
        n_jobs=-1,
        scoring='accuracy',
        verbose=2
    )
    grid.fit(X_train, y_train)

    print(f"Best params for {model_type}: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")

    # Return fitted best model
    return grid.best_estimator_, grid.best_params_