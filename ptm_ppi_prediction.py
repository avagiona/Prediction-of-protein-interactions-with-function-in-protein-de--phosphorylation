"""
PTM-PPI Prediction Framework
==============================
Python implementation of the machine learning pipeline from:

  Vagiona et al. (2025). "Prediction of protein interactions with function
  in protein (de-)phosphorylation." PLoS ONE 20(3): e0319084.
  https://doi.org/10.1371/journal.pone.0319084




# ──────────────────────────────────────────────────────────────────────────────
# 1.  Feature matrix construction
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "r_effector", "theta_effector",
    "DC_effector", "BC_effector", "CC_effector", "EC_effector",
    "r_target",   "theta_target",
    "DC_target",  "BC_target",  "CC_target",  "EC_target",
    "hyp_dist",   "r_diff",
]


def build_feature_matrix(edge_df: pd.DataFrame,
                          node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the 14-feature matrix used in the paper.

    Features (per the Methods section):
      • r, theta                – hyperbolic coordinates of effector & target
      • DC, BC, CC, EC          – four centrality measures for each node
      • hyp_dist                – hyperbolic distance of the edge
      • r_diff                  – |r_effector − r_target|
    """
    eff_feats = node_df.loc[edge_df["effector"]].reset_index(drop=True)
    tgt_feats = node_df.loc[edge_df["target"]].reset_index(drop=True)
    edge_info = edge_df[["hyp_dist", "r_diff"]].reset_index(drop=True)

    X = pd.DataFrame({
        "r_effector":    eff_feats["r"].values,
        "theta_effector":eff_feats["theta"].values,
        "DC_effector":   eff_feats["DC"].values,
        "BC_effector":   eff_feats["BC"].values,
        "CC_effector":   eff_feats["CC"].values,
        "EC_effector":   eff_feats["EC"].values,
        "r_target":      tgt_feats["r"].values,
        "theta_target":  tgt_feats["theta"].values,
        "DC_target":     tgt_feats["DC"].values,
        "BC_target":     tgt_feats["BC"].values,
        "CC_target":     tgt_feats["CC"].values,
        "EC_target":     tgt_feats["EC"].values,
        "hyp_dist":      edge_info["hyp_dist"].values,
        "r_diff":        edge_info["r_diff"].values,
    })
    return X


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Under-sampling to address class imbalance
# ──────────────────────────────────────────────────────────────────────────────

def undersample(X: pd.DataFrame, y: pd.Series,
                seed: int = 42) -> tuple:
    """
    Under-sample the majority class (non-PTM) to match the minority class,
    following the paper's approach during cross-validation.
    """
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index
    neg_sampled = resample(neg_idx, n_samples=len(pos_idx),
                           replace=False, random_state=seed)
    keep = pos_idx.union(neg_sampled)
    return X.loc[keep], y.loc[keep]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Random Forest model – train & evaluate
# ──────────────────────────────────────────────────────────────────────────────

class PTMPredictor:
    """
    Random Forest predictor for directed PTM-related protein interactions.

    Mirrors the paper's setup:
      • mtry  (max_features) = 14  (all features considered at each split)
      • ntrees               = 500
      • 5-fold CV repeated 10 times for hyperparameter selection
      • 70 / 30 train-test split
    """

    def __init__(self, n_estimators: int = 500,
                 max_features: int = 14,
                 random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            class_weight="balanced",   # alternative to explicit under-sampling
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_importances_ = None
        self.trained = False

    # ── fit ───────────────────────────────────────────────────────────────
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        self.trained = True
        return self

    # ── predict ───────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    # ── cross-validation ──────────────────────────────────────────────────
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                        n_splits: int = 5,
                        n_repeats: int = 10) -> dict:
        """
        5-fold CV repeated n_repeats times (paper: repeats = 10).
        Returns mean ± std of accuracy, ROC-AUC.
        """
        from sklearn.model_selection import RepeatedStratifiedKFold
        cv = RepeatedStratifiedKFold(n_splits=n_splits,
                                     n_repeats=n_repeats,
                                     random_state=42)
        scores = cross_validate(
            self.model, X, y,
            cv=cv,
            scoring=["accuracy", "roc_auc"],
            return_train_score=False,
            n_jobs=-1,
        )
        return {
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std":  scores["test_accuracy"].std(),
            "roc_auc_mean":  scores["test_roc_auc"].mean(),
            "roc_auc_std":   scores["test_roc_auc"].std(),
        }

    # ── full evaluation report ────────────────────────────────────────────
    def evaluate(self, X_test: pd.DataFrame,
                  y_test: pd.Series) -> dict:
        y_pred  = self.predict(X_test)
        y_score = self.predict_proba(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc      = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        avg_prec     = average_precision_score(y_test, y_score)
        cm           = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)   # recall
        specificity = tn / (tn + fp)

        return {
            "accuracy":    accuracy_score(y_test, y_pred),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "roc_auc":     roc_auc,
            "avg_precision": avg_prec,
            "fpr": fpr, "tpr": tpr,
            "precision": prec, "recall": rec,
            "confusion_matrix": cm,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Visualisation 
# ──────────────────────────────────────────────────────────────────────────────

def _style():
    """Apply a clean, publication-ready style."""
    plt.rcParams.update({
        "figure.dpi":      150,
        "figure.facecolor": "white",
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "font.family":     "DejaVu Sans",
        "font.size":       10,
        "axes.titlesize":  11,
        "axes.labelsize":  10,
    })


def plot_hyperbolic_map(node_df: pd.DataFrame,
                         edge_df: pd.DataFrame) -> plt.Figure:
    """
    Polar scatter plot of proteins in hyperbolic (H²) space.
    Effectors and targets are highlighted separately — reproduces Fig. 2A.
    """
    _style()
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="polar")

    ptm_edges = edge_df[edge_df["is_ptm"] == 1]
    effectors = set(ptm_edges["effector"])
    targets   = set(ptm_edges["target"])

    all_nodes = node_df.copy()
    all_nodes["role"] = "background"
    all_nodes.loc[all_nodes.index.isin(effectors), "role"] = "effector"
    all_nodes.loc[all_nodes.index.isin(targets),   "role"] = "target"

    colours = {"background": "#CCCCCC", "effector": "#E53935", "target": "#1E88E5"}
    sizes   = {"background": 5,          "effector": 30,        "target": 25}
    zorders = {"background": 1,          "effector": 3,         "target": 2}

    for role in ["background", "target", "effector"]:
        sub = all_nodes[all_nodes["role"] == role]
        ax.scatter(sub["theta"], sub["r"],
                   c=colours[role], s=sizes[role],
                   alpha=0.6, zorder=zorders[role], label=role.capitalize())

    ax.set_title("hPIN in Hyperbolic Space (H²)\n"
                 "Effectors & Targets of PTM Interactions",
                 pad=20, fontsize=12, fontweight="bold")
    ax.set_xlabel("Angular coordinate θ (functional similarity)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    return fig


def plot_feature_distributions(X: pd.DataFrame,
                                 y: pd.Series) -> plt.Figure:
    """
    Histogram grid comparing feature distributions for effectors,
    targets, and background nodes — reproduces Fig. 3A.
    """
    _style()
    features_to_plot = [
        "r_effector", "theta_effector",
        "DC_effector", "BC_effector",
        "CC_effector", "EC_effector",
        "hyp_dist",   "r_diff",
    ]
    labels = [f.replace("_effector", "").replace("_", " ").upper()
              for f in features_to_plot]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()

    positives = X[y == 1]
    negatives = X[y == 0]

    for ax, feat, lab in zip(axes, features_to_plot, labels):
        ax.hist(negatives[feat], bins=25, alpha=0.5,
                color="#AAAAAA", label="Background", density=True)
        ax.hist(positives[feat], bins=25, alpha=0.7,
                color="#E53935", label="PTM+", density=True)
        ax.set_title(lab)
        ax.set_xlabel("")
        ax.set_ylabel("Density")

    axes[0].legend(fontsize=8)
    fig.suptitle("Feature Distributions: PTM-positive vs Background",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_feature_importance(importances: pd.Series) -> plt.Figure:
    """Horizontal bar chart of feature importances — reproduces Fig. 3B."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1565C0" if "effector" in f else
              "#C62828" if "target" in f else "#2E7D32"
              for f in importances.index]
    importances.sort_values().plot.barh(ax=ax, color=colors[::-1])
    ax.set_xlabel("Feature Importance (mean decrease in impurity)")
    ax.set_title("Random Forest Feature Importances\n"
                 "(Blue = Effector  |  Red = Target  |  Green = Edge)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc_pr(metrics: dict) -> plt.Figure:
    """ROC and Precision-Recall curves — reproduces S2B & S2C Figs."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # ROC
    ax1.plot(metrics["fpr"], metrics["tpr"],
             color="#1565C0", lw=2,
             label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Receiver Operating Characteristic Curve", fontweight="bold")
    ax1.legend(loc="lower right")

    # Precision-Recall
    ax2.plot(metrics["recall"], metrics["precision"],
             color="#C62828", lw=2,
             label=f'PR (AP = {metrics["avg_precision"]:.3f})')
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve", fontweight="bold")
    ax2.legend(loc="upper right")

    fig.suptitle("Model Evaluation — PTM-PPI Random Forest Classifier",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    """Annotated confusion matrix."""
    _style()
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-PTM", "PTM"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix\n(30 % hold-out test set)",
                 fontweight="bold")
    plt.tight_layout()
    return fig


def plot_sca1_network(node_df: pd.DataFrame,
                       edge_df: pd.DataFrame,
                       predictor: PTMPredictor,
                       X: pd.DataFrame) -> plt.Figure:
    """
    Visualise a small disease sub-network centred on a 'hub' protein,
    analogous to the SCA1 ataxin-1 sub-network in Fig. 5A.

    Nodes are coloured by predicted PTM score.
    """
    _style()
    scores = predictor.predict_proba(X)
    top_ptm = edge_df.copy()
    top_ptm["score"] = scores
    top_ptm = top_ptm.nlargest(20, "score")

    G = nx.from_pandas_edgelist(top_ptm, "effector", "target",
                                  edge_attr="score")
    pos = nx.spring_layout(G, seed=42, k=0.6)

    fig, ax = plt.subplots(figsize=(9, 7))
    edge_weights = [d["score"] * 3 for _, _, d in G.edges(data=True)]
    node_colors  = [node_df.loc[n, "EC"] if n in node_df.index else 0.5
                    for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, width=edge_weights,
                           alpha=0.6, edge_color="#666666", ax=ax)
    nodes = nx.draw_networkx_nodes(G, pos,
                                    node_color=node_colors,
                                    cmap=plt.cm.RdYlBu_r,
                                    node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Eigenvector Centrality")

    ax.set_title("Top-Predicted PTM-PPI Sub-network\n"
                 "(analogous to SCA1 ataxin-1 cluster, Fig. 5A)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(n_proteins: int = 500,
                  n_ptm_interactions: int = 295,
                  test_size: float = 0.30,
                  seed: int = 42,
                  save_figures: bool = True) -> dict:
    """
    End-to-end reproduction of the paper's ML pipeline.

    Steps
    -----
    1. Generate / load the hPIN with hyperbolic coordinates
    2. Build 14-feature matrix
    3. Undersample majority class
    4. Train / test split  (70 / 30)
    5. 5-fold CV × 10 repeats
    6. Final model fit + evaluation on hold-out set
    7. Feature importance analysis
    8. Save all figures

    Returns
    -------
    dict with model, metrics, and dataframes.
    """
    print("=" * 60)
    print("  PTM-PPI Prediction Pipeline")
    print("  Vagiona et al., PLoS ONE (2025)")
    print("=" * 60)

    # ── 1. Data ─────────────────────────────────────────────────────────
    print("\n[1/6]  Generating synthetic hPIN …")
    node_df, edge_df, _ = generate_synthetic_hpin(
        n_proteins, n_ptm_interactions, seed=seed
    )
    print(f"       Proteins : {len(node_df):,}")
    print(f"       Edges    : {len(edge_df):,}")
    print(f"       PTM+     : {edge_df['is_ptm'].sum():,}")

    # ── 2. Features ──────────────────────────────────────────────────────
    print("\n[2/6]  Building 14-feature matrix …")
    X = build_feature_matrix(edge_df, node_df)
    y = edge_df["is_ptm"].reset_index(drop=True)
    print(f"       Shape: {X.shape}  |  features: {list(X.columns)}")

    # ── 3. Under-sampling ────────────────────────────────────────────────
    print("\n[3/6]  Under-sampling majority class …")
    X_bal, y_bal = undersample(X, y, seed=seed)
    print(f"       Balanced dataset: {X_bal.shape[0]} samples "
          f"({y_bal.sum()} PTM+, {(y_bal == 0).sum()} non-PTM)")

    # ── 4. Train / test split ────────────────────────────────────────────
    print("\n[4/6]  Splitting data (70/30) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=test_size,
        stratify=y_bal, random_state=seed
    )
    print(f"       Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── 5. Cross-validation ──────────────────────────────────────────────
    print("\n[5/6]  5-fold CV × 10 repeats …  (this may take a moment)")
    predictor = PTMPredictor(random_state=seed)
    cv_scores = predictor.cross_validate(X_train, y_train,
                                          n_splits=5, n_repeats=10)
    print(f"       CV Accuracy : {cv_scores['accuracy_mean']:.3f} "
          f"± {cv_scores['accuracy_std']:.3f}")
    print(f"       CV ROC-AUC  : {cv_scores['roc_auc_mean']:.3f} "
          f"± {cv_scores['roc_auc_std']:.3f}")

    # ── 6. Final fit + evaluation ────────────────────────────────────────
    print("\n[6/6]  Training final model and evaluating on hold-out set …")
    predictor.fit(X_train, y_train)
    metrics = predictor.evaluate(X_test, y_test)

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  Test Accuracy  : {metrics['accuracy']:.3f}            │")
    print(f"  │  Sensitivity    : {metrics['sensitivity']:.3f}            │")
    print(f"  │  Specificity    : {metrics['specificity']:.3f}            │")
    print(f"  │  ROC-AUC        : {metrics['roc_auc']:.3f}            │")
    print(f"  │  Avg Precision  : {metrics['avg_precision']:.3f}            │")
    print(f"  └─────────────────────────────────┘")
    print()
    print("  Classification report:")
    print(classification_report(
        y_test, predictor.predict(X_test),
        target_names=["Non-PTM", "PTM"]
    ))

    # ── Figures ──────────────────────────────────────────────────────────
    print("  Generating figures …")

    fig1 = plot_hyperbolic_map(node_df, edge_df)
    fig2 = plot_feature_distributions(X, y)
    fig3 = plot_feature_importance(predictor.feature_importances_)
    fig4 = plot_roc_pr(metrics)
    fig5 = plot_confusion_matrix(metrics["confusion_matrix"])
    fig6 = plot_sca1_network(node_df, edge_df, predictor, X_test)

    if save_figures:
        for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5, fig6], 1):
            fname = f"figure_{i}.png"
            fig.savefig(fname, bbox_inches="tight", dpi=150)
            print(f"    Saved {fname}")

    print("\n  Done.")
    return {
        "node_df":    node_df,
        "edge_df":    edge_df,
        "X":          X,
        "y":          y,
        "predictor":  predictor,
        "metrics":    metrics,
        "cv_scores":  cv_scores,
        "figures":    [fig1, fig2, fig3, fig4, fig5, fig6],
    }

