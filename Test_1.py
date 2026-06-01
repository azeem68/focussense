

import tensorflow as tf
from keras import layers, models, regularizers, callbacks
import kagglehub
import pandas as pd
import numpy as np
import joblib
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ─── 1. DOWNLOAD DATASETS ────────────────────────────────────────────────────
path_gaze = kagglehub.dataset_download("uwrfkaggler/ravdess-facial-landmark-tracking")
path_pose = kagglehub.dataset_download("mohamedadlyi/aflw2000-3d")

# ─── 2. DATA LOADING ─────────────────────────────────────────────────────────
def load_pose_data(path):
    """Load yaw, pitch, AND roll from .mat files."""
    data = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith('.mat'):
                mat = loadmat(os.path.join(root, f))
                if 'Pose_Para' in mat:
                    pose = mat['Pose_Para'][0]
                    # pose[0]=pitch, pose[1]=yaw, pose[2]=roll
                    pitch, yaw, roll = pose[0], pose[1], pose[2]
                    data.append([yaw, pitch, roll])
    return pd.DataFrame(data, columns=['yaw', 'pitch', 'roll'])


def load_gaze_data(path):
    """Load all CSV files and keep only numeric columns."""
    file_paths = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith('.csv'):
                file_paths.append(os.path.join(root, f))

    if not file_paths:
        print("No CSV files found!")
        return pd.DataFrame()

    dfs = []
    for fp in file_paths[:200]:
        try:
            dfs.append(pd.read_csv(fp))
        except Exception:
            continue

    df = pd.concat(dfs, ignore_index=True)
    # Keep only numeric columns to avoid pipeline issues
    return df.select_dtypes(include=[np.number])


# ─── 3. LOAD DATA ────────────────────────────────────────────────────────────
df_gaze = load_gaze_data(path_gaze)
df_pose = load_pose_data(path_pose)

print(f"Gaze rows: {len(df_gaze)} | Pose rows: {len(df_pose)}")
print(f"Gaze columns: {df_gaze.columns.tolist()[:10]} ...")

# ─── 4. FEATURE ENGINEERING & LABELS ────────────────────────────────────────

# --- Gaze ---
# Use all gaze_angle columns; also add AU (Action Unit) columns if present
gaze_cols = [c for c in df_gaze.columns if 'gaze' in c.lower()]
au_cols    = [c for c in df_gaze.columns if c.startswith('AU')]
feature_cols_gaze = gaze_cols + au_cols

if not feature_cols_gaze:
    # Fallback: use all numeric columns
    feature_cols_gaze = df_gaze.columns.tolist()

df_gaze_feat = df_gaze[feature_cols_gaze].copy()

# Label: distracted if gaze angle exceeds threshold
gaze_x_col = next((c for c in gaze_cols if 'angle_x' in c), None)
gaze_y_col = next((c for c in gaze_cols if 'angle_y' in c), None)

if gaze_x_col and gaze_y_col:
    df_gaze['target'] = (
        (df_gaze[gaze_x_col].abs() > 0.3) |
        (df_gaze[gaze_y_col].abs() > 0.3)
    ).astype(int)
else:
    # Fallback: label via median deviation of first two columns
    col0 = df_gaze_feat.columns[0]
    df_gaze['target'] = (df_gaze[col0].abs() > df_gaze[col0].std()).astype(int)

# Drop rows with NaN features or labels
df_gaze_feat['target'] = df_gaze['target']
df_gaze_feat.dropna(inplace=True)

X1 = df_gaze_feat.drop(columns='target').values
y1 = df_gaze_feat['target'].values

# --- Pose ---
# Use yaw, pitch, roll + derived magnitude feature
df_pose['magnitude'] = np.sqrt(
    df_pose['yaw']**2 + df_pose['pitch']**2 + df_pose['roll']**2
)
yaw_thresh   = df_pose['yaw'].std()
pitch_thresh = df_pose['pitch'].std()
roll_thresh  = df_pose['roll'].std()

df_pose['target'] = (
    (df_pose['yaw'].abs()   > yaw_thresh)   |
    (df_pose['pitch'].abs() > pitch_thresh) |
    (df_pose['roll'].abs()  > roll_thresh)
).astype(int)

X2 = df_pose[['yaw', 'pitch', 'roll', 'magnitude']].values
y2 = df_pose['target'].values

print(f"\nGaze — class balance: {np.bincount(y1)}")
print(f"Pose  — class balance: {np.bincount(y2)}")

# ─── 5. TRAIN / TEST SPLIT ───────────────────────────────────────────────────
X_train_gaze, X_test_gaze, y_train_gaze, y_test_gaze = train_test_split(
    X1, y1, test_size=0.2, shuffle=True, random_state=42, stratify=y1)

X_train_pose, X_test_pose, y_train_pose, y_test_pose = train_test_split(
    X2, y2, test_size=0.2, shuffle=True, random_state=42, stratify=y2)

# ─── 6. SCALE ────────────────────────────────────────────────────────────────
scaler1 = StandardScaler()
X_train_gaze = scaler1.fit_transform(X_train_gaze)
X_test_gaze  = scaler1.transform(X_test_gaze)

scaler2 = StandardScaler()
X_train_pose = scaler2.fit_transform(X_train_pose)
X_test_pose  = scaler2.transform(X_test_pose)

# ─── 7. CLASS WEIGHTS (handles imbalanced data) ──────────────────────────────
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

cw_gaze = get_class_weights(y_train_gaze)
cw_pose = get_class_weights(y_train_pose)
print(f"\nGaze class weights: {cw_gaze}")
print(f"Pose  class weights: {cw_pose}")

# ─── 8. CALLBACKS ────────────────────────────────────────────────────────────
def make_callbacks(model_name):
    return [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'{model_name}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
    ]

# ─── 9. IMPROVED MODELS ────────────────────────────────────────────────────
def build_gaze_model(input_dim):
    """
    Deeper network with moderate regularization.
    L2=0.001 (was 0.1 — that was causing heavy underfitting).
    Dropout=0.3 (was 0.5).
    3 hidden layers instead of 1.
    """
    inp = layers.Input(shape=(input_dim,))

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


def build_pose_model(input_dim):
    """
    Deeper network for pose (4 features: yaw, pitch, roll, magnitude).
    Lighter architecture since feature space is smaller.
    """
    inp = layers.Input(shape=(input_dim,))

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(16, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model

# ─── 10. TRAIN ───────────────────────────────────────────────────────────────
print("\n=== Training Gaze Model ===")
my_model1 = build_gaze_model(X_train_gaze.shape[1])
my_model1.summary()

history1 = my_model1.fit(
    X_train_gaze, y_train_gaze,
    validation_split=0.2,
    epochs=10,           # EarlyStopping will halt when accuracy plateaus
    batch_size=64,        # smaller batch = more gradient updates per epoch
    class_weight=cw_gaze,
    callbacks=make_callbacks('gaze_model'),
    verbose=1
)

print("\n=== Training Pose Model ===")
my_model2 = build_pose_model(X_train_pose.shape[1])
my_model2.summary()

history2 = my_model2.fit(
    X_train_pose, y_train_pose,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    class_weight=cw_pose,
    callbacks=make_callbacks('pose_model'),
    verbose=1
)

# ─── 11. EVALUATE ────────────────────────────────────────────────────────────
print("\n=== Final Test Evaluation ===")
gaze_results = my_model1.evaluate(X_test_gaze, y_test_gaze, verbose=0)
pose_results = my_model2.evaluate(X_test_pose, y_test_pose, verbose=0)

metric_names = my_model1.metrics_names
print("\nGaze Model:")
for name, val in zip(metric_names, gaze_results):
    print(f"  {name}: {val:.4f}")

print("\nPose Model:")
for name, val in zip(metric_names, pose_results):
    print(f"  {name}: {val:.4f}")

# ─── 12. SAVE ────────────────────────────────────────────────────────────────
my_model1.save('gaze_model.keras')
my_model2.save('pose_model.keras')
joblib.dump(scaler1, 'gaze_scaler.pkl')
joblib.dump(scaler2, 'pose_scaler.pkl')
# Also save feature column names so inference knows which columns to use
joblib.dump(feature_cols_gaze, 'gaze_feature_cols.pkl')

print(f"\nDone! Trained on {len(X_train_gaze) + len(X_train_pose)} samples.")
print("Saved: gaze_model.keras, pose_model.keras, gaze_scaler.pkl, pose_scaler.pkl, gaze_feature_cols.pkl")
