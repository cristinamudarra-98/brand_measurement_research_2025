import arviz as az
import pytensor.tensor as pt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import  StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
import json

from tabulate import tabulate




def plot_data(
    model_df,
    date_column="date",
    scale_data=False,
    plot_height=800,
    columns_to_plot=None,
    title=None,
):
    """
    Plots specified columns of a dataframe over time, with an option to scale the data.

    Parameters:
    - model_df: Pandas DataFrame with a date column and other numeric columns to plot.
    - date_column: The name of the column containing date information.
    - scale_data: Boolean flag indicating whether to scale the data (True) or not (False).
    - plot_height: The height of the plot in pixels.
    - columns_to_plot: List of column names to be plotted. If None, all columns except the date column are plotted.
    """
    model_df[date_column] = pd.to_datetime(model_df[date_column])

    if columns_to_plot is None:
        columns_to_plot = [col for col in model_df.columns if col != date_column]

    if scale_data:
        scaler = StandardScaler()
        columns_to_scale = [
            col
            for col in columns_to_plot
            if col in model_df.columns and col != date_column
        ]
        scaled_df = (
            model_df.copy()
        ) 
        scaled_df[columns_to_scale] = scaler.fit_transform(model_df[columns_to_scale])
    else:
        scaled_df = model_df

    fig = go.Figure()

    for column in columns_to_plot:
        if column in scaled_df.columns: 
            fig.add_trace(
                go.Scatter(
                    x=scaled_df[date_column],
                    y=scaled_df[column],
                    mode="lines",
                    name=column,
                )
            )

    fig.update_layout(
        title=f"{title} Plot of Selected Columns Over Time",
        xaxis_title="Date",
        yaxis_title="Values" if not scale_data else "Scaled Values",
        legend_title="Columns",
        height=plot_height,
    )

    fig.show()


def plot_correlation_matrix(df: pd.DataFrame, vars: list, title=None):
    filtered_df = df[vars]
    corr_matrix = filtered_df.corr(method="spearman")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(2 * len(vars), 2 * len(vars)), facecolor="white")
    cmap = sns.diverging_palette(10, 133, sep=80, n=7)
    if len(vars) > 10:
        annot_size = 25
        axis_fontsize = 25
    else:
        annot_size = 15
        axis_fontsize = 15
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_size, "color": "black"},
        xticklabels=vars,
        yticklabels=vars,
    )
    plt.gca().set_facecolor("white")
    plt.xticks(rotation=80, fontsize=axis_fontsize)
    plt.yticks(rotation=0, fontsize=axis_fontsize)
    plt.title(f"{title} Correlation Matrix", fontsize=50)
    plt.show()
    return plt


def geometric_adstock(x, alpha, l_max, normalize):
    """Vectorized geometric adstock transformation."""
    cycles = [
        pt.concatenate(tensor_list=[pt.zeros(shape=x.shape)[:i], x[: x.shape[0] - i]])
        for i in range(l_max)
    ]
    x_cycle = pt.stack(cycles)
    x_cycle = pt.transpose(x=x_cycle, axes=[1, 2, 0])
    w = pt.as_tensor_variable([pt.power(alpha, i) for i in range(l_max)])
    w = pt.transpose(w)[None, ...]
    w = w / pt.sum(w, axis=2, keepdims=True) if normalize else w
    return pt.sum(pt.mul(x_cycle, w), axis=2)



def evaluate_mmm_with_mediator(
    unscaled_y_train, unscaled_y_test, train_preds, y_pred,
    unscaled_y_mediator_train=None, unscaled_y_mediator_test=None,
    mediator_train_preds=None, mediator_y_pred=None
):
    """
    Evaluate both the main model and mediator model performance.
    Does not include hierarchy evaluation.
    """
    
    results = {}
    
    def calculate_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        r2_value = az.r2_score(y_true, y_pred)
        if hasattr(r2_value, 'iloc'):  
            r2 = float(r2_value.iloc[0])
        else:
            r2 = float(r2_value)
        
        nrmse = rmse / (np.max(y_true) - np.min(y_true))
        return {
            "rmse": float(rmse),
            "r2": r2,
            "nrmse": float(nrmse)
        }
    
    # Evaluate main model
    print("\n===== MAIN MODEL EVALUATION =====")
    train_metrics = calculate_metrics(unscaled_y_train, train_preds)
    test_metrics = calculate_metrics(unscaled_y_test, y_pred)
    
    print("=== Training Metrics ===")
    print(f"R-Squared = {train_metrics['r2']:.4f}")
    print(f"RMSE = {train_metrics['rmse']:.4f}")
    print(f"NRMSE = {train_metrics['nrmse']:.4f}")
    
    print("\n=== Test Metrics ===")
    print(f"R-Squared = {test_metrics['r2']:.4f}")
    print(f"RMSE = {test_metrics['rmse']:.4f}")
    print(f"NRMSE = {test_metrics['nrmse']:.4f}")
    
    results["main_model"] = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }
    
    # Evaluate mediator model if data is provided
    if (unscaled_y_mediator_train is not None and unscaled_y_mediator_test is not None and
        mediator_train_preds is not None and mediator_y_pred is not None):
        
        print("\n\n===== MEDIATOR MODEL EVALUATION =====")
        mediator_train_metrics = calculate_metrics(unscaled_y_mediator_train, mediator_train_preds)
        mediator_test_metrics = calculate_metrics(unscaled_y_mediator_test, mediator_y_pred)
        
        print("=== Training Metrics ===")
        print(f"R-Squared = {mediator_train_metrics['r2']:.4f}")
        print(f"RMSE = {mediator_train_metrics['rmse']:.4f}")
        print(f"NRMSE = {mediator_train_metrics['nrmse']:.4f}")
        
        print("\n=== Test Metrics ===")
        print(f"R-Squared = {mediator_test_metrics['r2']:.4f}")
        print(f"RMSE = {mediator_test_metrics['rmse']:.4f}")
        print(f"NRMSE = {mediator_test_metrics['nrmse']:.4f}")
        
        results["mediator_model"] = {
            "train_metrics": mediator_train_metrics,
            "test_metrics": mediator_test_metrics
        }
        
        print("\n\n===== MEDIATOR-MAIN MODEL RELATIONSHIP =====")
        from scipy.stats import pearsonr
        corr_pred, p_value_pred = pearsonr(mediator_y_pred, y_pred)
        print(f"Correlation between mediator and main model predictions: {corr_pred:.4f} (p={p_value_pred:.4f})")
        
        corr_true, p_value_true = pearsonr(unscaled_y_mediator_test, unscaled_y_test)
        print(f"Correlation between mediator and main model true values: {corr_true:.4f} (p={p_value_true:.4f})")
        
        results["relationship"] = {
            "prediction_correlation": corr_pred,
            "prediction_p_value": p_value_pred,
            "true_value_correlation": corr_true,
            "true_value_p_value": p_value_true
        }
    
    return results



def summarize_variable(trace, var_name):
    """Extract summary statistics for a variable from the posterior trace."""
    posterior = trace.posterior[var_name].values.flatten()
    
    hdi_interval = az.hdi(posterior, hdi_prob=0.94)
    
    summary = pd.DataFrame({
        'variable': [var_name],
        'mean': [posterior.mean()],
        'sd': [posterior.std()],
        'hdi_3%': [hdi_interval[0]],
        'hdi_97%': [hdi_interval[1]],
        'median': [np.median(posterior)],
        'positive_prob': [(posterior > 0).mean()]
    })
    
    return summary





def calculate_contributions(
    trace,
    X_train,
    original_paid_features,
    original_competitor_features,
    original_control_features,
    seasonality=True,
    intercept=True,
    trend=True,
    include_mediator=True
):
    """
    Calculate and adjust contributions from a PyMC trace for main + LT/ST mediators.

    Parameters
    ----------
    trace : PyMC trace object or arviz.InferenceData
        Trace object containing posterior samples.
    X_train : pandas.DataFrame
        Training data used for modeling.
    original_paid_features : list
        Paid media feature names.
    original_competitor_features : list
        Competitor feature names.
    original_control_features : list
        Control feature names.
    seasonality : bool, default=True
        Whether to include seasonality component.
    intercept : bool, default=True
        Whether to include intercept.
    trend : bool, default=True
        Whether to include trend.
    include_mediator : bool, default=True
        Whether to include LT/ST mediator effects.

    Returns
    -------
    unadj_contributions : pandas.DataFrame
        Unadjusted contributions (raw values).
    adj_contributions : pandas.DataFrame
        Normalized (absolute %) contributions.
    """

    unadj_contributions = pd.DataFrame(index=X_train.index)

    # --- PAID ---
    for paid in original_paid_features:
        unadj_contributions[paid] = (
            trace["posterior"]["paid_contributions"]
            .sel(paid=paid)
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )

    # --- COMPETITOR ---
    for competitor in original_competitor_features:
        unadj_contributions[competitor] = (
            trace["posterior"]["competitor_contributions"]
            .sel(competitor=competitor)
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )

    # --- CONTROL ---
    for control in original_control_features:
        unadj_contributions[control] = (
            trace["posterior"]["control_contributions"]
            .sel(control=control)
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )

    # --- MEDIATOR EFFECTS (LT and ST) ---
    if include_mediator:
        # Long-term mediator
        if (
            "beta_mediator_lt_effect" in trace["posterior"]
            and "mu_mediator_lt" in trace["posterior"]
        ):
            mediator_lt_effect_value = float(trace["posterior"]["beta_mediator_lt_effect"].mean().values)
            mediator_lt_values = (
                trace["posterior"]["mu_mediator_lt"].mean(axis=1).mean(axis=0).to_numpy()
            )
            unadj_contributions["mediator_lt_effect"] = mediator_lt_effect_value * mediator_lt_values

        # Short-term mediator
        if (
            "beta_mediator_st_effect" in trace["posterior"]
            and "mu_mediator_st" in trace["posterior"]
        ):
            mediator_st_effect_value = float(trace["posterior"]["beta_mediator_st_effect"].mean().values)
            mediator_st_values = (
                trace["posterior"]["mu_mediator_st"].mean(axis=1).mean(axis=0).to_numpy()
            )
            unadj_contributions["mediator_st_effect"] = mediator_st_effect_value * mediator_st_values

    # --- SEASONALITY ---
    if seasonality and "seasonality" in trace["posterior"]:
        unadj_contributions["seasonality"] = (
            trace["posterior"]["seasonality"].mean(axis=1).mean(axis=0).to_numpy()
        )

    # --- INTERCEPT ---
    if intercept and "intercept" in trace["posterior"]:
        unadj_contributions["intercept"] = (
            trace["posterior"]["intercept"].mean(axis=1).mean(axis=0).to_numpy()
        )

    # --- TREND ---
    if trend and "trend" in trace["posterior"]:
        unadj_contributions["trend"] = (
            trace["posterior"]["trend"].mean(axis=1).mean(axis=0).to_numpy()
        )

    adj_contributions = unadj_contributions.abs().div(
        unadj_contributions.abs().sum(axis=1), axis=0
    )

    for competitor in original_competitor_features:
        if competitor in adj_contributions.columns:
            adj_contributions[competitor] *= -1

    if include_mediator:
        if "mediator_lt_effect" in adj_contributions.columns:
            mediator_lt_sign = np.sign(
                float(trace["posterior"]["beta_mediator_lt_effect"].mean().values)
            )
            if mediator_lt_sign < 0:
                adj_contributions["mediator_lt_effect"] *= -1

        if "mediator_st_effect" in adj_contributions.columns:
            mediator_st_sign = np.sign(
                float(trace["posterior"]["beta_mediator_st_effect"].mean().values)
            )
            if mediator_st_sign < 0:
                adj_contributions["mediator_st_effect"] *= -1

    return unadj_contributions, adj_contributions


def calculate_mediator_contributions(
    trace,
    X_train,
    mediator_lt_paid_features,
    mediator_lt_competitor_features,
    mediator_lt_control_features,
    mediator_st_paid_features,
    mediator_st_competitor_features,
    mediator_st_control_features,
    seasonality=True,
    intercept=True,
    trend=True,
):
    """
    Calculate and adjust contributions for LT and ST mediator models
    based on the given PyMC trace.

    Parameters
    ----------
    trace : arviz.InferenceData or dict
        Trace containing posterior draws from the mediator models.
    X_train : pandas.DataFrame
        Training data (used only for indexing).
    mediator_lt_*_features, mediator_st_*_features : list
        Lists of feature names per mediator and feature group.
    seasonality : bool, default=True
        Whether to include mediator seasonality if available.
    intercept : bool, default=True
        Whether to include mediator intercept.
    trend : bool, default=True
        Whether to include mediator trend.

    Returns
    -------
    mediator_lt_unadj, mediator_lt_adj, mediator_st_unadj, mediator_st_adj : pd.DataFrame
        Unadjusted and normalized (absolute %) contributions for both mediators.
    """

    def _extract_contributions(prefix, paid_feats, competitor_feats, control_feats):
        """Extract one mediator’s contributions based on prefix and feature names."""
        unadj = pd.DataFrame(index=X_train.index)

        # --- PAID ---
        if f"{prefix}_paid_contributions" in trace["posterior"]:
            arr = trace["posterior"][f"{prefix}_paid_contributions"]
            for paid in paid_feats:
                unadj[paid] = (
                    arr.sel({f"{prefix}_paid": paid})
                    .mean(axis=1)
                    .mean(axis=0)
                    .to_numpy()
                )

        # --- COMPETITOR ---
        if f"{prefix}_competitor_contributions" in trace["posterior"]:
            arr = trace["posterior"][f"{prefix}_competitor_contributions"]
            for comp in competitor_feats:
                unadj[comp] = (
                    arr.sel({f"{prefix}_competitor": comp})
                    .mean(axis=1)
                    .mean(axis=0)
                    .to_numpy()
                )

        # --- CONTROL ---
        if f"{prefix}_control_contributions" in trace["posterior"]:
            arr = trace["posterior"][f"{prefix}_control_contributions"]
            for ctrl in control_feats:
                unadj[ctrl] = (
                    arr.sel({f"{prefix}_control": ctrl})
                    .mean(axis=1)
                    .mean(axis=0)
                    .to_numpy()
                )

        # --- SEASONALITY ---
        if seasonality and f"{prefix}_seasonality" in trace["posterior"]:
            unadj["seasonality"] = (
                trace["posterior"][f"{prefix}_seasonality"]
                .mean(axis=1)
                .mean(axis=0)
                .to_numpy()
            )

        # --- INTERCEPT ---
        if intercept and f"{prefix}_intercept" in trace["posterior"]:
            unadj["intercept"] = (
                trace["posterior"][f"{prefix}_intercept"]
                .mean(axis=1)
                .mean(axis=0)
                .to_numpy()
            )

        # --- TREND ---
        if trend and f"{prefix}_trend" in trace["posterior"]:
            unadj["trend"] = (
                trace["posterior"][f"{prefix}_trend"]
                .mean(axis=1)
                .mean(axis=0)
                .to_numpy()
            )

        adj = unadj.abs().div(unadj.abs().sum(axis=1), axis=0)

        for comp in competitor_feats:
            if comp in adj.columns:
                adj[comp] *= -1

        return unadj, adj

    # === LONG-TERM MEDIATOR ===
    mediator_lt_unadj, mediator_lt_adj = _extract_contributions(
        prefix="mediator_lt",
        paid_feats=mediator_lt_paid_features,
        competitor_feats=mediator_lt_competitor_features,
        control_feats=mediator_lt_control_features,
    )

    # === SHORT-TERM MEDIATOR ===
    mediator_st_unadj, mediator_st_adj = _extract_contributions(
        prefix="mediator_st",
        paid_feats=mediator_st_paid_features,
        competitor_feats=mediator_st_competitor_features,
        control_feats=mediator_st_control_features,
    )

    return mediator_lt_unadj, mediator_lt_adj, mediator_st_unadj, mediator_st_adj


def plot_contributions(df, keep_intercept_trend_season=True):
    """
    Plot mean percentage contributions of each feature.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of contributions over time (each column = feature).
    keep_intercept_trend_season : bool, default=True
        Whether to include baseline components like intercept, trend, seasonality.

    Returns
    -------
    contributions_df : pandas.DataFrame
        DataFrame with mean and percentage contribution of each feature.
    fig : plotly.graph_objects.Figure
        Bar plot of percentage contributions.
    """
    df = df.copy()

    # Drop baseline components if requested
    baseline_cols = ["intercept", "trend", "seasonality"]
    if not keep_intercept_trend_season:
        cols_to_drop = [c for c in baseline_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")

    # --- Compute mean contribution over time ---
    mean_contributions = df.mean()

    # --- Compute absolute percentage contribution ---
    abs_sum = mean_contributions.abs().sum()
    pct_contrib = (mean_contributions.abs() / abs_sum) * 100

    # --- Preserve sign of contribution ---
    signed_contrib = pct_contrib * mean_contributions.apply(lambda x: 1 if x >= 0 else -1)

    # --- Build tidy DataFrame for output and plotting ---
    contributions_df = pd.DataFrame({
        "Feature": mean_contributions.index,
        "Mean": mean_contributions.values,
        "Percentage": signed_contrib.values
    }).sort_values(by="Percentage")

    # --- Plot ---
    fig = px.bar(
        contributions_df,
        x="Percentage",
        y="Feature",
        orientation="h",
        text="Percentage",
        labels={"Percentage": "Percentage Contribution (%)", "Feature": "Feature"},
        title="Average Contribution by Feature"
    )

    fig.update_traces(texttemplate="%{x:.2f}%", textposition="outside")
    fig.update_layout(
        height=600,
        width=1500,
    ) 

    return contributions_df, fig



def plot_normalized_coefficients(trace, var_names=None, include_mediator=True, figsize=(14, 10)):
    """
    Plots normalized coefficients from a PyMC trace for main model and LT/ST mediators.

    Parameters
    ----------
    trace : arviz.InferenceData
        Trace object containing posterior samples
    var_names : list, optional
        Variables to include; if None, inferred automatically.
    include_mediator : bool
        Whether to include mediator models.
    figsize : tuple
        Figure size for the plotly chart.
    """

    if not isinstance(trace, az.InferenceData):
        trace = az.from_pymc(trace)

    # --- Collect variables ---
    if var_names is None:
        main_vars = {
            var for var in trace.posterior.data_vars
            if var.startswith("beta_")
            and "fourier" not in var
            and "offset" not in var
            and not var.startswith("beta_mediator_")
        }

        # mediator effects
        for eff in ["beta_mediator_lt_effect", "beta_mediator_st_effect"]:
            if eff in trace.posterior.data_vars:
                main_vars.add(eff)

        mediator_lt_vars, mediator_st_vars = set(), set()
        if include_mediator:
            mediator_lt_vars = {
                var for var in trace.posterior.data_vars
                if var.startswith("mediator_lt_beta_")
                and "fourier" not in var
                and "offset" not in var
            }
            mediator_st_vars = {
                var for var in trace.posterior.data_vars
                if var.startswith("mediator_st_beta_")
                and "fourier" not in var
                and "offset" not in var
            }

        var_names = list(main_vars | mediator_lt_vars | mediator_st_vars)

    print(f"Variables to process: {var_names}")

    summary_data = []

    # --- Extract coefficients ---
    for var in var_names:
        # handle mediator effects
        if var in ["beta_mediator_lt_effect", "beta_mediator_st_effect"]:
            mean_value = float(trace.posterior[var].mean().values)
            model_type = "Main"
            label = "Mediator LT Effect" if "lt" in var else "Mediator ST Effect"
            summary_data.append({
                "Variable": label,
                "Model": model_type,
                "Coefficient": mean_value,
                "AbsCoefficient": abs(mean_value)
            })
            continue

        # handle vector parameters
        if hasattr(trace.posterior[var], "dims") and len(trace.posterior[var].dims) > 1:
            dim_name = var.replace("beta_", "").replace("_coeffs", "")
            if var.startswith("mediator_lt_beta_"):
                dim_name = var.replace("mediator_lt_beta_", "").replace("_coeffs", "")
            elif var.startswith("mediator_st_beta_"):
                dim_name = var.replace("mediator_st_beta_", "").replace("_coeffs", "")

            if dim_name in trace.posterior.coords:
                feature_names = trace.posterior.coords[dim_name].values
            else:
                feature_names = range(trace.posterior[var].shape[-1])

            for i, feature in enumerate(feature_names):
                try:
                    if dim_name in trace.posterior[var].dims:
                        mean_value = trace.posterior[var].sel({dim_name: feature}).mean().values
                    else:
                        mean_value = trace.posterior[var].values[:, :, i].mean()

                    if var.startswith("mediator_lt_"):
                        model_type = "Mediator_LT"
                    elif var.startswith("mediator_st_"):
                        model_type = "Mediator_ST"
                    else:
                        model_type = "Main"

                    summary_data.append({
                        "Variable": str(feature),
                        "Model": model_type,
                        "Coefficient": float(mean_value),
                        "AbsCoefficient": abs(float(mean_value))
                    })
                except Exception as e:
                    print(f"Error processing {var} - {feature}: {e}")
        else:
            # scalar parameter
            try:
                mean_value = float(trace.posterior[var].mean().values)
                if var.startswith("mediator_lt_"):
                    model_type = "Mediator_LT"
                elif var.startswith("mediator_st_"):
                    model_type = "Mediator_ST"
                else:
                    model_type = "Main"

                summary_data.append({
                    "Variable": var.replace("beta_", "").replace("mediator_", ""),
                    "Model": model_type,
                    "Coefficient": mean_value,
                    "AbsCoefficient": abs(mean_value)
                })
            except Exception as e:
                print(f"Error processing variable {var}: {e}")

    # --- Build DataFrame ---
    coeffs_df = pd.DataFrame(summary_data)
    print(f"Extracted {len(coeffs_df)} coefficients from {len(var_names)} variables.")

    # --- Normalize by absolute sum within each model ---
    normalized_dfs = []
    for model in coeffs_df["Model"].unique():
        model_df = coeffs_df[coeffs_df["Model"] == model].copy()
        total_abs = model_df["AbsCoefficient"].sum()
        if total_abs == 0:
            model_df["Normalized"] = 0
        else:
            model_df["Normalized"] = (
                model_df["AbsCoefficient"] / total_abs * 100 *
                model_df["Coefficient"].apply(lambda x: 1 if x >= 0 else -1)
            )
        normalized_dfs.append(model_df)

    final_df = pd.concat(normalized_dfs)
    final_df = final_df.sort_values(["Model", "Normalized"], ascending=[True, True])

    print(f"Final normalized coefficients table:\n{final_df.head()}")

    # --- Plot ---
    fig = px.bar(
        final_df,
        x="Normalized",
        y="Variable",
        color="Model",
        orientation="h",
        text="Normalized",
        barmode="group",
        height=figsize[1] * 50,
        width=figsize[0] * 50,
        labels={"Normalized": "Relative Importance (%)"},
        title="Normalized Coefficients for Main, LT Mediator, and ST Mediator Models"
    )

    fig.update_layout(
        xaxis_title="Relative Importance (%)",
        yaxis_title="Features",
        legend_title="Model",
        font=dict(size=12)
    )
    fig.update_traces(texttemplate="%{x:.2f}%", textposition="outside")

    return fig


def get_coefficient_table(trace, include_mediator=True):
    """
    Create a summary table of coefficients for the main, LT mediator, and ST mediator models.

    Parameters
    ----------
    trace : arviz.InferenceData
        Trace object containing posterior samples
    include_mediator : bool, default=True
        Whether to include mediator model coefficients

    Returns
    -------
    pandas.DataFrame
        Formatted table of coefficients with Model, Variable, Feature, Mean, HDI, SD, P(>0)
    """
    print("All available variables:", list(trace.posterior.data_vars))

    # --- MAIN MODEL VARIABLES ---
    main_vars = [
        var for var in trace.posterior.data_vars
        if var.startswith("beta_")
        and "fourier" not in var
        and "offset" not in var
        and not var.startswith("beta_mediator_")
    ]

    # Ensure mediator effects are included in the main model
    for eff in ["beta_mediator_lt_effect", "beta_mediator_st_effect"]:
        if eff not in main_vars and eff in trace.posterior.data_vars:
            main_vars.append(eff)

    # --- MEDIATOR VARIABLES ---
    mediator_lt_vars, mediator_st_vars = [], []
    if include_mediator:
        mediator_lt_vars = [
            var for var in trace.posterior.data_vars
            if var.startswith("mediator_lt_beta_")
            and "fourier" not in var
            and "offset" not in var
        ]
        mediator_st_vars = [
            var for var in trace.posterior.data_vars
            if var.startswith("mediator_st_beta_")
            and "fourier" not in var
            and "offset" not in var
        ]

    all_vars = main_vars + mediator_lt_vars + mediator_st_vars
    results = []

    for var in all_vars:
        # --- Handle mediator effects ---
        if var in ["beta_mediator_lt_effect", "beta_mediator_st_effect"]:
            slice_data = trace.posterior[var].values.flatten()
            hdi_interval = az.hdi(slice_data, hdi_prob=0.94)
            row = {
                "Model": "Main",
                "Variable": var,
                "Feature": "Mediator LT Effect" if "lt" in var else "Mediator ST Effect",
                "Mean": slice_data.mean(),
                "HDI 3%": hdi_interval[0],
                "HDI 97%": hdi_interval[1],
                "SD": slice_data.std(),
                "P(>0)": (slice_data > 0).mean()
            }
            results.append(row)
            continue

        # --- Handle regular coefficients ---
        try:
            coeff_values = trace.posterior[var]
            if hasattr(coeff_values, "dims") and len(coeff_values.dims) > 1:
                # Determine coordinate name
                dim_name = var.replace("beta_", "").replace("_coeffs", "")
                if var.startswith("mediator_lt_beta_"):
                    dim_name = var.replace("mediator_lt_beta_", "").replace("_coeffs", "")
                elif var.startswith("mediator_st_beta_"):
                    dim_name = var.replace("mediator_st_beta_", "").replace("_coeffs", "")

                # Get feature names
                if dim_name in trace.posterior.coords:
                    feature_names = trace.posterior.coords[dim_name].values
                else:
                    # fallback if coords missing
                    feature_names = range(coeff_values.shape[-1])

                for i, feature in enumerate(feature_names):
                    slice_data = (
                        coeff_values.sel({dim_name: feature}).values.flatten()
                        if dim_name in coeff_values.dims
                        else coeff_values.values[:, :, i].flatten()
                    )
                    hdi_interval = az.hdi(slice_data, hdi_prob=0.94)

                    # Determine model type
                    if var.startswith("mediator_lt_"):
                        model_type = "Mediator_LT"
                    elif var.startswith("mediator_st_"):
                        model_type = "Mediator_ST"
                    else:
                        model_type = "Main"

                    row = {
                        "Model": model_type,
                        "Variable": var,
                        "Feature": str(feature),
                        "Mean": slice_data.mean(),
                        "HDI 3%": hdi_interval[0],
                        "HDI 97%": hdi_interval[1],
                        "SD": slice_data.std(),
                        "P(>0)": (slice_data > 0).mean()
                    }
                    results.append(row)
            else:
                # Scalar parameter
                slice_data = coeff_values.values.flatten()
                hdi_interval = az.hdi(slice_data, hdi_prob=0.94)
                if var.startswith("mediator_lt_"):
                    model_type = "Mediator_LT"
                elif var.startswith("mediator_st_"):
                    model_type = "Mediator_ST"
                else:
                    model_type = "Main"
                row = {
                    "Model": model_type,
                    "Variable": var,
                    "Feature": var,
                    "Mean": slice_data.mean(),
                    "HDI 3%": hdi_interval[0],
                    "HDI 97%": hdi_interval[1],
                    "SD": slice_data.std(),
                    "P(>0)": (slice_data > 0).mean()
                }
                results.append(row)

        except Exception as e:
            print(f"Error processing variable {var}: {e}")

    # --- Final formatting ---
    df = pd.DataFrame(results)
    numeric_cols = ["Mean", "HDI 3%", "HDI 97%", "SD", "P(>0)"]
    df[numeric_cols] = df[numeric_cols].applymap(lambda x: round(x, 3) if pd.notnull(x) else x)
    df = df.sort_values(["Model", "Variable"])

    print("\n=== Coefficient Summary Table ===")
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

    return df



def compare_coefficients(trace):
    """
    Create a side-by-side comparison of main model vs both mediator models (LT & ST).
    Focuses on matched coefficients (like paid media, organic media, etc.)

    Parameters
    ----------
    trace : arviz.InferenceData
        Trace object containing posterior samples

    Returns
    -------
    pandas.DataFrame
        Formatted comparison table with Main, Mediator LT, and Mediator ST coefficients
    """
    all_coeffs = get_coefficient_table(trace, include_mediator=True)

    main_coeffs = all_coeffs[all_coeffs['Model'] == 'Main'].copy()
    mediator_lt_coeffs = all_coeffs[all_coeffs['Model'] == 'Mediator_LT'].copy()
    mediator_st_coeffs = all_coeffs[all_coeffs['Model'] == 'Mediator_ST'].copy()

    mediator_effects = all_coeffs[all_coeffs['Variable'].str.contains("beta_mediator_")]
    if not mediator_effects.empty:
        print("\n=== Mediator Effects on Main Outcome ===")
        mediator_effects_formatted = mediator_effects[['Variable', 'Mean', 'HDI 3%', 'HDI 97%', 'P(>0)']].copy()
        print(tabulate(mediator_effects_formatted, headers='keys', tablefmt='psql', showindex=False))

    print("\n=== Main Model vs Mediator Models (LT & ST) Coefficients ===")

    def extract_type(var, model_prefix):
        if 'beta_' in var and '_coeffs' in var:
            parts = var.split('_')
            try:
                if model_prefix == 'Main':
                    return parts[1]  
                elif model_prefix == 'Mediator_LT':
                    return parts[2]  
                elif model_prefix == 'Mediator_ST':
                    return parts[2]
            except IndexError:
                return 'other'
        return 'other'

    for df, prefix in [
        (main_coeffs, 'Main'),
        (mediator_lt_coeffs, 'Mediator_LT'),
        (mediator_st_coeffs, 'Mediator_ST')
    ]:
        df['Type'] = df['Variable'].apply(lambda x: extract_type(x, prefix))

    comparison_rows = []
    feature_types = (
        set(main_coeffs['Type'])
        .union(set(mediator_lt_coeffs['Type']))
        .union(set(mediator_st_coeffs['Type']))
    )
    feature_types = [ft for ft in feature_types if ft != 'other']

    for ftype in feature_types:
        main_type = main_coeffs[main_coeffs['Type'] == ftype]
        lt_type = mediator_lt_coeffs[mediator_lt_coeffs['Type'] == ftype]
        st_type = mediator_st_coeffs[mediator_st_coeffs['Type'] == ftype]
        features = set(main_type['Feature']).union(set(lt_type['Feature'])).union(set(st_type['Feature']))

        for feature in features:
            main_row = main_type[main_type['Feature'] == feature]
            lt_row = lt_type[lt_type['Feature'] == feature]
            st_row = st_type[st_type['Feature'] == feature]

            row = {
                'Type': ftype.capitalize(),
                'Feature': feature,
                'Main Mean': main_row['Mean'].values[0] if not main_row.empty else None,
                'Main HDI': f"[{main_row['HDI 3%'].values[0]:.3f}, {main_row['HDI 97%'].values[0]:.3f}]" if not main_row.empty else None,
                'Main P(>0)': main_row['P(>0)'].values[0] if not main_row.empty else None,

                'Mediator LT Mean': lt_row['Mean'].values[0] if not lt_row.empty else None,
                'Mediator LT HDI': f"[{lt_row['HDI 3%'].values[0]:.3f}, {lt_row['HDI 97%'].values[0]:.3f}]" if not lt_row.empty else None,
                'Mediator LT P(>0)': lt_row['P(>0)'].values[0] if not lt_row.empty else None,

                'Mediator ST Mean': st_row['Mean'].values[0] if not st_row.empty else None,
                'Mediator ST HDI': f"[{st_row['HDI 3%'].values[0]:.3f}, {st_row['HDI 97%'].values[0]:.3f}]" if not st_row.empty else None,
                'Mediator ST P(>0)': st_row['P(>0)'].values[0] if not st_row.empty else None
            }
            comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(['Type', 'Feature'])
        print(tabulate(comparison_df, headers='keys', tablefmt='psql', showindex=False))

    return comparison_df



def plot_mmm_time_series(
    train_index, test_index,
    main_y_train, main_y_test, main_train_preds, main_test_preds,
    
    # Mediator LT
    mediator_lt_y_train=None, mediator_lt_y_test=None,
    mediator_lt_train_preds=None, mediator_lt_test_preds=None,
    mediator_lt_y_label="Mediator LT",
    
    # Mediator ST
    mediator_st_y_train=None, mediator_st_y_test=None,
    mediator_st_train_preds=None, mediator_st_test_preds=None,
    mediator_st_y_label="Mediator ST",
    
    main_y_label="Main Target",
    show_plots=True):
    """
    Plot time series of predictions vs actuals for the main model and up to two mediators (LT and ST).

    Parameters
    ----------
    train_index, test_index : array-like
        Date indices for training and test sets.
    main_y_train, main_y_test : array-like
        Actual values for the main model.
    main_train_preds, main_test_preds : array-like
        Predicted values for the main model.
    mediator_lt_y_train, mediator_lt_y_test : array-like, optional
        Actual values for the long-term mediator.
    mediator_lt_train_preds, mediator_lt_test_preds : array-like, optional
        Predicted values for the long-term mediator.
    mediator_st_y_train, mediator_st_y_test : array-like, optional
        Actual values for the short-term mediator.
    mediator_st_train_preds, mediator_st_test_preds : array-like, optional
        Predicted values for the short-term mediator.
    main_y_label, mediator_lt_y_label, mediator_st_y_label : str
        Y-axis labels for each model.
    show_plots : bool
        Whether to display the plots.
    save_path : str, optional
        Path prefix to save the figures (e.g., 'results/mmm').
    """
    
    figures = {}

    def plot_single_series(index_train, index_test, y_train, y_test, pred_train, pred_test, label_prefix):
        f_train = plt.figure(figsize=(12, 8))
        plt.title(f"{label_prefix} | Training Set")
        plt.plot(index_train, y_train, label="Actual", marker='o')
        plt.plot(index_train, pred_train, label="Predicted", marker='x')
        plt.xlabel("Date")
        plt.ylabel(label_prefix)
        plt.legend()
        plt.grid(True, alpha=0.3)
        figures[f"{label_prefix.lower().replace(' ', '_')}_train"] = f_train

        f_test = plt.figure(figsize=(12, 8))
        plt.title(f"{label_prefix} | Test Set")
        plt.plot(index_test, y_test, label="Actual", marker='o')
        plt.plot(index_test, pred_test, label="Predicted", marker='x')
        plt.xlabel("Date")
        plt.ylabel(label_prefix)
        plt.legend()
        plt.grid(True, alpha=0.3)
        figures[f"{label_prefix.lower().replace(' ', '_')}_test"] = f_test

    # === MAIN MODEL ===
    plot_single_series(train_index, test_index, main_y_train, main_y_test, main_train_preds, main_test_preds, main_y_label)

    # === MEDIATOR LT ===
    if all(v is not None for v in [mediator_lt_y_train, mediator_lt_y_test, mediator_lt_train_preds, mediator_lt_test_preds]):
        plot_single_series(train_index, test_index, mediator_lt_y_train, mediator_lt_y_test, mediator_lt_train_preds, mediator_lt_test_preds, mediator_lt_y_label)

    # === MEDIATOR ST ===
    if all(v is not None for v in [mediator_st_y_train, mediator_st_y_test, mediator_st_train_preds, mediator_st_test_preds]):
        plot_single_series(train_index, test_index, mediator_st_y_train, mediator_st_y_test, mediator_st_train_preds, mediator_st_test_preds, mediator_st_y_label)

    if show_plots:
        plt.show()

    return figures


def plot_adstock_effects(
    trace,
    feature_type="paid",
    include_mediator=True,
    l_max=16,
    impulse_value=100,
    height=800,
    width=1200
):
    """
    Plot cumulative adstock effects for any feature type in both main model and mediator model.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        ArviZ trace object from PyMC sampling
    feature_type : str, default="paid"
        Type of features to plot adstock for: "paid" or "organic"
    include_mediator : bool, default=True
        Whether to include mediator model visualization  
    l_max : int, default=16
        Maximum lag for the adstock effect
    impulse_value : float, default=100
        Value to use for the initial impulse
    height, width : int
        Plot dimensions
        
    Returns:
    --------
    dict
        Dictionary of Plotly figure objects for main and mediator models
    """

    
    if feature_type not in ["paid", "organic"]:
        raise ValueError("feature_type must be 'paid' or 'organic'")
    
    main_alpha_var = f"alpha_{feature_type}"
    mediator_alpha_var = f"mediator_alpha_{feature_type}"
    main_features = list(trace.posterior.coords[feature_type].values)
    
    figures = {}
    main_alpha = az.summary(trace, var_names=main_alpha_var, round_to=3)['mean'].values
    main_data = pd.DataFrame({
        channel: [impulse_value] + [0] * (l_max - 1) 
        for channel in main_features
    })
    main_data[[feature + '_adstocked' for feature in main_features]] = \
        geometric_adstock(x=main_data[main_features], alpha=main_alpha, l_max=l_max, normalize=True).eval()
    for channel in main_features:
        adstocked_column = channel + '_adstocked'
        cumulative_column = channel + '_cumulative'
        main_data[cumulative_column] = main_data[adstocked_column].cumsum()
    
    main_fig = go.Figure()
    for i, channel in enumerate(main_features):
        cumulative_column = channel + '_cumulative'
        main_fig.add_trace(
            go.Scatter(
                x=main_data.index, 
                y=main_data[cumulative_column], 
                mode='lines+markers', 
                name=f"{channel} (α={main_alpha[i]:.3f})",
                line=dict(width=3),
                marker=dict(size=8)
            )
        )
    
    title_type = "Marketing Channels" if feature_type == "paid" else "Campaign Activities"
    main_fig.update_layout(
        title=f'Main Model: Cumulative Adstock Effects for {title_type}',
        xaxis_title='Weeks',
        yaxis_title='Cumulative Adstock Effect',
        legend_title=feature_type.capitalize(),
        template='plotly_white',
        height=height,
        width=width
    )
    
    main_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    main_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    figures['main'] = main_fig
    
    if include_mediator:
        try:
            mediator_features = list(trace.posterior.coords[f"mediator_{feature_type}"].values)
            mediator_alpha = az.summary(trace, var_names=mediator_alpha_var, round_to=3)['mean'].values
            mediator_data = pd.DataFrame({
                channel: [impulse_value] + [0] * (l_max - 1) 
                for channel in mediator_features
            })
            mediator_data[[feature + '_adstocked' for feature in mediator_features]] = \
                geometric_adstock(x=mediator_data[mediator_features], alpha=mediator_alpha, l_max=l_max, normalize=True).eval()
            for channel in mediator_features:
                adstocked_column = channel + '_adstocked'
                cumulative_column = channel + '_cumulative'
                mediator_data[cumulative_column] = mediator_data[adstocked_column].cumsum()
            mediator_fig = go.Figure()
            for i, channel in enumerate(mediator_features):
                cumulative_column = channel + '_cumulative'
                mediator_fig.add_trace(
                    go.Scatter(
                        x=mediator_data.index, 
                        y=mediator_data[cumulative_column], 
                        mode='lines+markers', 
                        name=f"{channel} (α={mediator_alpha[i]:.3f})",
                        line=dict(width=3),
                        marker=dict(size=8)
                    )
                )
            mediator_fig.update_layout(
                title=f'Mediator Model: Cumulative Adstock Effects for {title_type}',
                xaxis_title='Weeks',
                yaxis_title='Cumulative Adstock Effect',
                legend_title=feature_type.capitalize(),
                template='plotly_white',
                height=height,
                width=width
            )
            mediator_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            mediator_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            figures['mediator'] = mediator_fig
            
        except (KeyError, ValueError) as e:
            print(f"Could not plot mediator model: {str(e)}")
    
    return figures




def plot_combined_adstock_effects(
    trace,
    feature_type="paid",
    l_max=16,
    impulse_value=100,
    height=900,
    width=1200
):
    """
    Plot adstock effects for both models in a single figure with subplots.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        ArviZ trace object from PyMC sampling
    feature_type : str, default="paid"
        Type of features to plot adstock for: "paid" or "organic"
    l_max : int, default=16
        Maximum lag for the adstock effect
    impulse_value : float, default=100
        Value to use for the initial impulse
    height, width : int
        Plot dimensions
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Combined figure with subplots
    """
    if feature_type not in ["paid", "organic"]:
        raise ValueError("feature_type must be 'paid' or 'organic'")
    
    main_alpha_var = f"alpha_{feature_type}"
    mediator_alpha_var = f"mediator_alpha_{feature_type}"
    main_features = list(trace.posterior.coords[feature_type].values)
    mediator_features = list(trace.posterior.coords[f"mediator_{feature_type}"].values)
    title_type = "Marketing Channels" if feature_type == "paid" else "Campaign Activities"
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Main Model: Cumulative Adstock Effects for {title_type}',
            f'Mediator Model: Cumulative Adstock Effects for {title_type}'
        ),
        vertical_spacing=0.15
    )
    
    main_alpha = az.summary(trace, var_names=main_alpha_var, round_to=3)['mean'].values
    mediator_alpha = az.summary(trace, var_names=mediator_alpha_var, round_to=3)['mean'].values
    main_data = pd.DataFrame({
        channel: [impulse_value] + [0] * (l_max - 1) 
        for channel in main_features
    })
    
    main_data[[feature + '_adstocked' for feature in main_features]] = \
        geometric_adstock(x=main_data[main_features], alpha=main_alpha, l_max=l_max, normalize=True).eval()
    
    for channel in main_features:
        adstocked_column = channel + '_adstocked'
        cumulative_column = channel + '_cumulative'
        main_data[cumulative_column] = main_data[adstocked_column].cumsum()
    mediator_data = pd.DataFrame({
        channel: [impulse_value] + [0] * (l_max - 1) 
        for channel in mediator_features
    })
    
    mediator_data[[feature + '_adstocked' for feature in mediator_features]] = \
        geometric_adstock(x=mediator_data[mediator_features], alpha=mediator_alpha, l_max=l_max, normalize=True).eval()
    
    for channel in mediator_features:
        adstocked_column = channel + '_adstocked'
        cumulative_column = channel + '_cumulative'
        mediator_data[cumulative_column] = mediator_data[adstocked_column].cumsum()
    for i, channel in enumerate(main_features):
        cumulative_column = channel + '_cumulative'
        fig.add_trace(
            go.Scatter(
                x=main_data.index, 
                y=main_data[cumulative_column], 
                mode='lines+markers', 
                name=f"Main: {channel} (α={main_alpha[i]:.3f})",
                line=dict(width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    for i, channel in enumerate(mediator_features):
        cumulative_column = channel + '_cumulative'
        fig.add_trace(
            go.Scatter(
                x=mediator_data.index, 
                y=mediator_data[cumulative_column], 
                mode='lines+markers', 
                name=f"Mediator: {channel} (α={mediator_alpha[i]:.3f})",
                line=dict(width=3, dash='dot'),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
    feature_name = "Marketing" if feature_type == "paid" else "Campaign" 
    fig.update_layout(
        title=f'Comparison of Adstock Effects: Main vs Mediator Model ({feature_name})',
        template='plotly_white',
        height=height,
        width=width,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    fig.update_xaxes(title_text="Weeks", row=2, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    fig.update_yaxes(title_text="Cumulative Adstock Effect", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Adstock Effect", row=2, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig





def plot_contributions_over_time(
    unadj_contributions,
    plot_order=None,
    title="Predicted Watchtime and Breakdown",
    colors_file_path=None,
    height=1200,
    width=1500
):
    """
    Plot contribution breakdown over time with optional custom colors.
    
    Parameters:
    -----------
    unadj_contributions : pandas.DataFrame
        DataFrame with unadjusted contributions
    plot_order : list, optional
        Specific order for plotting certain features first (if None, will use alphabetical order)
    title : str
        Plot title
    colors_file_path : str, optional
        Path to JSON file with color mappings
    height, width : int
        Plot dimensions
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with stacked area chart
    """
    color_dict = {"positive_colors": {}, "negative_colors": {}}
    if colors_file_path:
        color_dict = load_color_mappings(colors_file_path)

    if plot_order is None:
        plot_order = []

    positive_parts = unadj_contributions.where(unadj_contributions > 0)
    positive_parts.dropna(axis=1, how='all', inplace=True)

    negative_parts = unadj_contributions.where(unadj_contributions < 0)
    negative_parts.dropna(axis=1, how='all', inplace=True)

    fig = go.Figure()

    ordered_columns = plot_order + [c for c in positive_parts.columns if c not in plot_order]

    for col in ordered_columns:
        if col in positive_parts.columns:
            color = color_dict.get("positive_colors", {}).get(col, None)
            fig.add_trace(go.Scatter(
                x=positive_parts.index,
                y=positive_parts[col],
                stackgroup='one',
                name=col,
                mode='lines',
                line=dict(width=0.5, color=color),
                fill='tonexty',
                fillcolor=color
            ))

    for col in negative_parts.columns:
        color = color_dict.get("negative_colors", {}).get(col, None)
        fig.add_trace(go.Scatter(
            x=negative_parts.index,
            y=negative_parts[col],
            stackgroup='two',
            name=col,
            mode='lines',
            line=dict(width=0.5, color=color),
            fill='tonexty',
            fillcolor=color
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Watchtime Contribution",
        hovermode="x unified",
        height=height, 
        width=width,
    )

    return fig



def load_color_mappings(color_file_path):
    """
    Load color mappings from a JSON file
    
    Args:
        color_file_path: Path to the JSON color mapping file
        
    Returns:
        Dictionary with color mappings
    """
    try:
        with open(color_file_path, 'r') as f:
            color_dict = json.load(f)
        return color_dict
    except Exception as e:
        print(f"Error loading color mappings: {e}")
        return {"positive_colors": {}, "negative_colors": {}}
    
    
    
def evaluate_lt_st_mmm_with_mediator(
    # Main model
    unscaled_y_train, unscaled_y_test, train_preds, y_pred,
    
    # Mediator LT
    unscaled_y_mediator_lt_train=None, unscaled_y_mediator_lt_test=None,
    mediator_lt_train_preds=None, mediator_lt_y_pred=None,
    
    # Mediator ST
    unscaled_y_mediator_st_train=None, unscaled_y_mediator_st_test=None,
    mediator_st_train_preds=None, mediator_st_y_pred=None
):
    """
    Evaluate both the main model and up to two mediator model performances.
    Includes correlations between mediators and the main model.
    """
    
    results = {}

    def calculate_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2_value = az.r2_score(y_true, y_pred)
        r2 = float(r2_value.iloc[0]) if hasattr(r2_value, 'iloc') else float(r2_value)
        nrmse = rmse / (np.max(y_true) - np.min(y_true))
        return {"rmse": float(rmse), "r2": r2, "nrmse": float(nrmse)}

    def evaluate_single_model(model_name, y_train, y_test, preds_train, preds_test):
        print(f"\n===== {model_name.upper()} EVALUATION =====")
        train_metrics = calculate_metrics(y_train, preds_train)
        test_metrics = calculate_metrics(y_test, preds_test)

        print("=== Training Metrics ===")
        print(f"R-Squared = {train_metrics['r2']:.4f}")
        print(f"RMSE = {train_metrics['rmse']:.4f}")
        print(f"NRMSE = {train_metrics['nrmse']:.4f}")

        print("\n=== Test Metrics ===")
        print(f"R-Squared = {test_metrics['r2']:.4f}")
        print(f"RMSE = {test_metrics['rmse']:.4f}")
        print(f"NRMSE = {test_metrics['nrmse']:.4f}")

        return {"train_metrics": train_metrics, "test_metrics": test_metrics}

    def evaluate_relationship(mediator_name, y_mediator_test, mediator_pred_test):
        corr_pred, p_pred = pearsonr(mediator_pred_test, y_pred)
        corr_true, p_true = pearsonr(y_mediator_test, unscaled_y_test)
        print(f"\nCorrelation between {mediator_name} and main model predictions: {corr_pred:.4f} (p={p_pred:.4f})")
        print(f"Correlation between {mediator_name} and main model true values: {corr_true:.4f} (p={p_true:.4f})")
        return {
            "prediction_correlation": corr_pred,
            "prediction_p_value": p_pred,
            "true_value_correlation": corr_true,
            "true_value_p_value": p_true
        }

    # === MAIN MODEL ===
    results["main_model"] = evaluate_single_model(
        "Main Model", unscaled_y_train, unscaled_y_test, train_preds, y_pred
    )

    # === MEDIATOR LT ===
    if all(v is not None for v in [
        unscaled_y_mediator_lt_train, unscaled_y_mediator_lt_test,
        mediator_lt_train_preds, mediator_lt_y_pred
    ]):
        results["mediator_lt_model"] = evaluate_single_model(
            "Mediator LT Model",
            unscaled_y_mediator_lt_train, unscaled_y_mediator_lt_test,
            mediator_lt_train_preds, mediator_lt_y_pred
        )
        print("\n===== LT-MAIN MODEL RELATIONSHIP =====")
        results["relationship_lt"] = evaluate_relationship(
            "Mediator LT", unscaled_y_mediator_lt_test, mediator_lt_y_pred
        )

    # === MEDIATOR ST ===
    if all(v is not None for v in [
        unscaled_y_mediator_st_train, unscaled_y_mediator_st_test,
        mediator_st_train_preds, mediator_st_y_pred
    ]):
        results["mediator_st_model"] = evaluate_single_model(
            "Mediator ST Model",
            unscaled_y_mediator_st_train, unscaled_y_mediator_st_test,
            mediator_st_train_preds, mediator_st_y_pred
        )
        print("\n===== ST-MAIN MODEL RELATIONSHIP =====")
        results["relationship_st"] = evaluate_relationship(
            "Mediator ST", unscaled_y_mediator_st_test, mediator_st_y_pred
        )

    return results


def plot_combined_adstock_effects(
    trace,
    feature_type,              # one of: "paid", "mediator_lt_paid", "mediator_st_paid"
    l_max=16,
    impulse_value=100,
    height=600,
    width=1200,
):
    """
    Plot cumulative adstock effects for a given feature group (main, mediator LT, or mediator ST).

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior trace.
    feature_type : str
        Must be one of {"paid", "mediator_lt_paid", "mediator_st_paid"}.
    l_max : int
        Adstock window length.
    impulse_value : float
        Size of the impulse applied at t=0.
    height, width : int
        Figure dimensions.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    # Map feature group -> alpha var and coord name
    alpha_var_map = {
        "paid": "alpha_paid",
        "mediator_lt_paid": "mediator_lt_alpha_paid",
        "mediator_st_paid": "mediator_st_alpha_paid",
    }
    coord_map = {
        "paid": "paid",
        "mediator_lt_paid": "mediator_lt_paid",
        "mediator_st_paid": "mediator_st_paid",
    }

    if feature_type not in alpha_var_map:
        raise ValueError(
            "feature_type must be one of: 'paid', 'mediator_lt_paid', 'mediator_st_paid'"
        )

    alpha_var = alpha_var_map[feature_type]
    coord_name = coord_map[feature_type]

    # Validate presence in trace
    if alpha_var not in trace.posterior:
        raise KeyError(f"Alpha variable '{alpha_var}' not found in trace.posterior.")
    if coord_name not in trace.posterior.coords:
        raise KeyError(f"Coord '{coord_name}' not found in trace.posterior.coords.")

    # Get feature names
    features = list(trace.posterior.coords[coord_name].values)

    # Get posterior-mean alpha vector
    alpha_da = trace.posterior[alpha_var]
    if {"chain", "draw"}.issubset(alpha_da.dims):
        alpha_mean = alpha_da.mean(dim=("chain", "draw"))
    else:
        reduce_dims = [d for d in alpha_da.dims if d != coord_name]
        alpha_mean = alpha_da.mean(dim=reduce_dims) if reduce_dims else alpha_da

    if coord_name in alpha_mean.dims:
        alpha_vec = alpha_mean.sel({coord_name: features}).to_numpy()
    else:
        arr = np.asarray(alpha_mean)
        alpha_vec = np.repeat(float(arr), len(features)) if arr.ndim == 0 else arr

    # Create impulse (100 at t=0)
    impulse_df = pd.DataFrame({f: [impulse_value] + [0] * (l_max - 1) for f in features})

    # Compute adstock and cumulative response
    adstocked = geometric_adstock(
        x=impulse_df[features],
        alpha=alpha_vec,
        l_max=l_max,
        normalize=True
    ).eval()

    cumulative = np.cumsum(adstocked, axis=0)

    # Title mapping
    title_map = {
        "paid": "Main Model – Paid Media",
        "mediator_lt_paid": "Mediator LT – Paid Media",
        "mediator_st_paid": "Mediator ST – Paid Media",
    }

    # Plot cumulative adstock effects
    fig = go.Figure()
    for i, ch in enumerate(features):
        fig.add_trace(
            go.Scatter(
                x=np.arange(l_max),
                y=cumulative[:, i],
                mode="lines+markers",
                name=ch
            )
        )

    fig.update_layout(
        title=f"Cumulative Adstock Effects ({title_map[feature_type]})",
        xaxis_title="Weeks (Lag)",
        yaxis_title="Cumulative Adstock Effect",
        legend_title="Feature",
        template="plotly_white",
        height=height,
        width=width
    )

    return fig
