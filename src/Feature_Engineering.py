import numpy as np
import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, IncrementalPCA
import umap.umap_ as umap
import matplotlib.pyplot as plt
def feature_engineering():
    df="/Users/salonisahal/Stock-Price-Prediction/Data/preprocessed_data.csv"
    df = pd.read_csv(df)


    PCA_COMPONENTS = 3
    UMAP_COMPONENTS = 2
    UMAP_SAMPLE_SIZE = 20000       # fit UMAP on this many rows max
    PLOT_SAMPLE = 5000             # plot only this many points to avoid renderer overload
    IPCA_BATCH_SIZE = 5000         # batch size for IncrementalPCA (tune to memory)

# ---------------------------
# Prepare numeric feature matrix (drop non-numeric / label cols)
# ---------------------------
# Make a copy to avoid modifying original df accidentally
    _work = df.copy()

# Ensure date and id/label columns are removed
    cols_to_drop = ["date", "target", "Target_Movement", "stock_id"]
    cols_to_drop = [c for c in cols_to_drop if c in _work.columns]
    X_df = _work.drop(columns=cols_to_drop)

# Keep only numeric columns (safe)
    X_df = X_df.select_dtypes(include=[np.number])

# ---------------------------
# Impute missing values and remove zero-variance columns
# ---------------------------
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_df)   # numpy array

# Remove zero-variance features (VarianceThreshold)
    vt = VarianceThreshold(threshold=0.0)
    X_non_const = vt.fit_transform(X_imputed)

# ---------------------------
# Scale features
# ---------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_non_const)

# Free memory
    del X_imputed
    gc.collect()


# PCA: use IncrementalPCA if dataset is large

    n_samples = X_scaled.shape[0]
    if n_samples > 50000:
        print("Using IncrementalPCA (large dataset).")
        ipca = IncrementalPCA(n_components=PCA_COMPONENTS, batch_size=IPCA_BATCH_SIZE)
        X_pca = ipca.fit_transform(X_scaled)
    else:
        print("Using standard PCA.")
        pca = PCA(n_components=PCA_COMPONENTS)
        X_pca = pca.fit_transform(X_scaled)

# Build PCA dataframe columns names
    pca_cols = [f"PCA{i+1}" for i in range(PCA_COMPONENTS)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=_work.index)
    _work = pd.concat([_work, df_pca], axis=1)

# If you want explained variance (only available for standard PCA)
    try:
        print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    except Exception:
        try:
            print("Explained Variance Ratio (Incremental PCA): approx not available directly")
        except:
            pass


# UMAP: fit on a sample, then transform whole dataset

# Choose sample indices (deterministic)
    sample_size = min(UMAP_SAMPLE_SIZE, n_samples)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

# Fit UMAP on sampled scaled data
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=UMAP_COMPONENTS, random_state=42)
    print(f"Fitting UMAP on {sample_size} samples (this may take a while)...")
    umap_model.fit(X_scaled[sample_idx])

# Transform the full dataset (faster + lower memory than fit on whole)
    X_umap = umap_model.transform(X_scaled)
    umap_cols = [f"UMAP{i+1}" for i in range(UMAP_COMPONENTS)]
    df_umap = pd.DataFrame(X_umap, columns=umap_cols, index=_work.index)
    _work = pd.concat([_work, df_umap], axis=1)

# Free memory
    del X_scaled, X_pca, X_umap
    gc.collect()


# Plot (sampled for safety)

    plot_sample = min(PLOT_SAMPLE, n_samples)
    plot_idx = rng.choice(n_samples, size=plot_sample, replace=False)

    plt.figure(figsize=(8,6))
    sc = plt.scatter(_work.loc[_work.index[plot_idx], "PCA1"],
                 _work.loc[_work.index[plot_idx], "PCA2"],
                 c=_work.loc[_work.index[plot_idx], "Target_Movement"],
                 cmap="viridis", s=8, alpha=0.8)
    plt.colorbar(sc, label="Target_Movement")
    plt.xlabel("PCA1"); plt.ylabel("PCA2"); plt.title("PCA visualization (sampled)")
    plt.show()

    plt.figure(figsize=(8,6))
    sc2 = plt.scatter(_work.loc[_work.index[plot_idx], "UMAP1"],
                  _work.loc[_work.index[plot_idx], "UMAP2"],
                  c=_work.loc[_work.index[plot_idx], "Target_Movement"],
                  cmap="viridis", s=8, alpha=0.8)
    plt.colorbar(sc2, label="Target_Movement")
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2"); plt.title("UMAP visualization (sampled)")
    plt.show()


# Assign back to original df (if you want)

# If you want to keep these features in your main dataframe
    df_with_embeddings = _work  # contains original columns + PCA*/UMAP*
# (optionally) df = df_with_embeddings.copy()
    df_with_embeddings.to_csv("/Users/salonisahal/Stock-Price-Prediction/Data/featured_data.csv", index=False)
    return