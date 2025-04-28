import gc
import sys
import math
import joblib
import logging
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from pytorch_tabnet.tab_model import TabNetRegressor

# Hyperopt imports
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

"""to train for all the conditions, a loop should be added to make it more automated but its alright for now I guess"""


# 1. PREPARE DATA

def prepare_data(return_data=False, dependent_file_type='', input_file_path=f'/ibex/project/c2205/sara/klebsiella_pneumoniae/unitigs_kp.rtab'):

    # Read the input rtab file
    df = pd.read_csv(input_file_path, delimiter='\t', index_col=0)
    reference_df = pd.read_csv(f'/ibex/user/hinkovn/alg_comparison/gene_presence_absence.Rtab', delimiter='\t', index_col=0)

    # Transpose for samples as rows
    X = df.T
    reference_df = reference_df.T

    # Align X with the reference (same index)
    X = X.reindex(reference_df.index)

    del df
    gc.collect()

    # Read the target CSV (median measurements, etc.)
    y = pd.read_csv(f'/ibex/user/hinkovn/TabNet/dependent_median_{dependent_file_type}_ML.csv')

    # Force same index as reference
    y.index = reference_df.index

    del reference_df
    gc.collect()

    if return_data:
        input_features = X.columns
        output_features = y.columns
        data = pd.concat([X, y], axis=1)
        return data, input_features, output_features

    return X, y


# CALL PREPARE_DATA

X, y = prepare_data(
    dependent_file_type='opacity',  
    input_file_path='/ibex/project/c2205/sara/klebsiella_pneumoniae/unitigs_kp.rtab'
)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Pick your target column from y
target_column = "Gentamycin_4ugml"
y_target = y[target_column]


# 2. SPLIT INTO TRAIN & TEST SETS (HOLD-OUT TEST SET)

TrainX, TestX, TrainY, TestY = train_test_split(
    X, y_target,
    test_size=0.2,
    random_state=313
)


# 3. OPTIONAL - APPLY A PCA 

n_components = 500
pca = PCA(n_components=n_components, svd_solver='arpack')
TrainX_pca = pca.fit_transform(TrainX)
TestX_pca  = pca.transform(TestX)


# joblib.dump(pca, "pca_transformer.pkl")

# Make arrays contiguous to avoid negative-stride issues
TrainX_pca = np.ascontiguousarray(TrainX_pca)
TestX_pca  = np.ascontiguousarray(TestX_pca)

TrainY_vals = TrainY.values.reshape(-1, 1)  # TabNet wants 2D
TestY_vals  = TestY.values.reshape(-1, 1)

# 4. HYPERPARAMETER TUNING (K-fold CV on the *Train set* with Hyperopt)

def hyperopt_objective_tabnet(params):
    """
    Objective function for Hyperopt that:
      - Does K-fold cross-validation on (TrainX_pca, TrainY_vals)
      - Returns average validation RMSE from the final epoch
    """
    # Convert discrete parameters
    params['n_steps'] = int(params['n_steps'])
    lr = params.pop('lr')  # We'll pass this to optimizer_params

    # 3-fold CV
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    cv_scores = []
    for train_idx, val_idx in kf.split(TrainX_pca):
        X_tr, X_val = TrainX_pca[train_idx], TrainX_pca[val_idx]
        y_tr, y_val = TrainY_vals[train_idx], TrainY_vals[val_idx]

        model = TabNetRegressor(
            **params,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': lr},
            # For example, a simple step LR schedule (optional):
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            seed=42
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=['rmse'],
            max_epochs=300,   # You can tune or shorten for speed
            patience=20,      # Early stopping
            batch_size=256,
            virtual_batch_size=128
        )
        # 'val_0_rmse' is the validation set of the first (and only) eval_set
        # The final entry in model.history['val_0_rmse'] is the last epoch's RMSE
        final_rmse = model.history['val_0_rmse'][-1]
        cv_scores.append(final_rmse)

    avg_rmse = np.mean(cv_scores)
    return {'loss': avg_rmse, 'status': STATUS_OK}

# Define a Hyperopt search space
space_tabnet = {
    'n_d': hp.choice('n_d', [16, 24, 32]),
    'n_a': hp.choice('n_a', [16, 24, 32]),
    'n_steps': hp.quniform('n_steps', 3, 6, 1),
    'gamma': hp.quniform('gamma', 1.0, 1.5, 0.1),
    'lambda_sparse': hp.loguniform('lambda_sparse', math.log(1e-5), math.log(1e-3)),
    'lr': hp.loguniform('lr', math.log(1e-4), math.log(1e-1)),  # learning rate
    'mask_type': hp.choice('mask_type', ["entmax", "sparsemax"])
}

print("\nStarting TabNet hyperparameter optimization with Hyperopt...")
trials_tabnet = Trials()
best_tabnet = fmin(
    fn=hyperopt_objective_tabnet,
    space=space_tabnet,
    algo=tpe.suggest,
    max_evals=300,  # Increase for a more exhaustive search
    trials=trials_tabnet
)
best_params_tabnet = space_eval(space_tabnet, best_tabnet)
best_params_tabnet['n_steps'] = int(best_params_tabnet['n_steps'])
print(f"\nBest TabNet parameters: {best_params_tabnet}")


# 5. TRAIN FINAL MODEL ON THE ENTIRE TRAIN SET WITH BEST HYPERPARAMS

best_lr = best_params_tabnet.pop('lr')  
final_model = TabNetRegressor(
    **best_params_tabnet,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr': best_lr},
    seed=42
)

final_model.fit(
    TrainX_pca, TrainY_vals,
    eval_set=[(TestX_pca, TestY_vals)],
    eval_metric=['rmse'],
    max_epochs=500,   # You can increase epochs for the final model
    patience=20,
    batch_size=128,
    virtual_batch_size=64
)

# Evaluate on test set
test_rmse = final_model.history['val_0_rmse'][-1]
print(f"\nFinal model RMSE on test set: {test_rmse:.4f}")

# Save the trained model
joblib.dump(final_model, f"TabNet_PCA_opacity_{target_column}.pkl")
print(f"Model saved")
