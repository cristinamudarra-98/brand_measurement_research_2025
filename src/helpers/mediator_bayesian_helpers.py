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
    original_organic_features,
    original_control_features,
    seasonality=True,
    intercept=True,
    trend=True,
    include_mediator=True
):
    """
    Calculate and adjust contributions from a PyMC trace.

    Parameters:
    trace: PyMC trace object
    X_train: DataFrame with training data
    original_paid_features: List of paid media features
    original_competitor_features: List of competitor features
    original_organic_features: List of organic features
    original_control_features: List of control features
    seasonality, intercept, trend: Boolean flags for including these components
    include_mediator: Boolean flag for including mediator effect

    Returns:
    unadj_contributions, adj_contributions: DataFrames of contributions
    """
    unadj_contributions = pd.DataFrame(index=X_train.index)

    for paid in original_paid_features:
        unadj_contributions[paid] = (
            trace["posterior"]["paid_contributions"]
            .sel(paid=paid)
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    for competitor in original_competitor_features:
        unadj_contributions[competitor] = (
            trace["posterior"]["competitor_contributions"]
            .sel(competitor=competitor)
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    for organic in original_organic_features:
        unadj_contributions[organic] = (
            trace["posterior"]["organic_contributions"]
            .sel(organic=organic)
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    for control in original_control_features:
        unadj_contributions[control] = (
            trace["posterior"]["control_contributions"]
            .sel(control=control)
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    if include_mediator and "beta_mediator_effect" in trace["posterior"] and "mu_mediator" in trace["posterior"]:
        mediator_effect_value = float(trace["posterior"]["beta_mediator_effect"].mean().values)
        mediator_values = trace["posterior"]["mu_mediator"].mean(axis=1).mean(axis=0).to_numpy()
        unadj_contributions["mediator_effect"] = mediator_effect_value * mediator_values
    
    if seasonality and "seasonality" in trace["posterior"]:
        unadj_contributions["seasonality"] = (
            trace["posterior"]["seasonality"]
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    if intercept:
        unadj_contributions["intercept"] = (
            trace["posterior"]["intercept"]
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    if trend and "trend" in trace["posterior"]:
        unadj_contributions["trend"] = (
            trace["posterior"]["trend"]
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )

    adj_contributions = unadj_contributions.abs().div(
        unadj_contributions.abs().sum(axis=1), axis=0
    )
    
    for competitor in original_competitor_features:
        if competitor in adj_contributions.columns:
            adj_contributions[competitor] *= -1
    
    if include_mediator and "mediator_effect" in adj_contributions.columns:
        mediator_sign = np.sign(float(trace["posterior"]["beta_mediator_effect"].mean().values))
        if mediator_sign < 0:
            adj_contributions["mediator_effect"] *= -1

    return unadj_contributions, adj_contributions


def calculate_mediator_contributions(
    trace,
    X_train,
    mediator_paid_features,
    mediator_competitor_features,
    mediator_organic_features,
    mediator_control_features,
    seasonality=True,
    intercept=True,
    trend=True,
):
    """
    Calculate and adjust contributions from a PyMC trace for mediator model.

    Parameters:
    trace: PyMC trace object
    X_train: DataFrame with training data
    mediator_paid_features: List of mediator paid media features
    mediator_competitor_features: List of mediator competitor features
    mediator_organic_features: List of mediator organic features
    mediator_control_features: List of mediator control features
    seasonality, intercept, trend: Boolean flags for including these components

    Returns:
    mediator_unadj_contributions, mediator_adj_contributions: DataFrames of mediator contributions
    """
    mediator_unadj_contributions = pd.DataFrame(index=X_train.index)
    
    for paid in mediator_paid_features:
        mediator_unadj_contributions[paid] = (
            trace["posterior"]["mediator_paid_contributions"]
            .sel(mediator_paid=paid)
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    for competitor in mediator_competitor_features:
        mediator_unadj_contributions[competitor] = (
            trace["posterior"]["mediator_competitor_contributions"]
            .sel(mediator_competitor=competitor)
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    for organic in mediator_organic_features:
        mediator_unadj_contributions[organic] = (
            trace["posterior"]["mediator_organic_contributions"]
            .sel(mediator_organic=organic)
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    for control in mediator_control_features:
        mediator_unadj_contributions[control] = (
            trace["posterior"]["mediator_control_contributions"]
            .sel(mediator_control=control)
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    if seasonality:
        mediator_unadj_contributions["seasonality"] = (
            trace["posterior"]["mediator_seasonality"]
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    if intercept:
        mediator_unadj_contributions["intercept"] = (
            trace["posterior"]["mediator_intercept"]
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    if trend:
        mediator_unadj_contributions["trend"] = (
            trace["posterior"]["mediator_trend"]
            .mean(axis=1)
            .mean(axis=0)
            .to_numpy()
        )
    
    mediator_adj_contributions = mediator_unadj_contributions.abs().div(
        mediator_unadj_contributions.abs().sum(axis=1), axis=0
    )
    
    for competitor in mediator_competitor_features:
        mediator_adj_contributions[competitor] *= -1
    
    return mediator_unadj_contributions, mediator_adj_contributions


def plot_contributions(df, keep_intercept_trend_season=True):
    """
    Plot contributions over time.

    Parameters:
    df: DataFrame containing the data to plot
    keep_intercept_trend_season: Boolean choice to keep or exclude baseline components

    Returns:
    contributions_df, fig: DataFrame of the contributions and the figure
    """
    if not keep_intercept_trend_season:
        df = df.drop(columns=["intercept", "trend", "seasonality"])
        
    mean_contributions = pd.DataFrame(df.mean())
    mean_contributions.columns = ["mean"]
    contributions = (
        mean_contributions.abs().mean(axis=1)
        / mean_contributions.abs().mean(axis=1).sum()
        * 100
        * mean_contributions["mean"].apply(lambda x: 1 if x >= 0 else -1)
    )
    contributions_df = pd.DataFrame(contributions, columns=["Contribution"])
    contributions_df["Features"] = contributions_df.index
    fig = px.bar(
        contributions_df.sort_values(by="Contribution"),
        x="Contribution",
        y="Features",
        orientation="h",
        labels={"Contribution": "Percentage Contribution", "Features": "Features"},
        title="Contribution of Features",
    )
    fig.update_layout(
        height=600,
        width=1500,
    ) 
    fig.update_traces(textangle=0, texttemplate="%{x:.2f}")
    return contributions_df, fig



def plot_normalized_coefficients(trace, var_names=None, include_mediator=True, figsize=(14, 10)):
    """
    Plots normalized coefficients from a PyMC trace for both main model and mediator.
    """

    if not isinstance(trace, az.InferenceData):
        trace = az.from_pymc(trace)
    
    if var_names is None:
        main_vars = set([var for var in trace.posterior.data_vars 
                     if var.startswith('beta_') and 'fourier' not in var and 'offset' not in var])
        
        if 'beta_mediator_effect' in trace.posterior.data_vars:
            main_vars.add('beta_mediator_effect')
            
        if include_mediator:
            mediator_vars = [var for var in trace.posterior.data_vars 
                            if var.startswith('mediator_beta_') and 'fourier' not in var and 'offset' not in var]
            var_names = list(main_vars) + mediator_vars
        else:
            var_names = list(main_vars)
    
    print(f"Variables to process: {var_names}")
    summary_data = []
    for var in var_names:
        if not include_mediator and var.startswith('mediator_'):
            continue
        if var == 'beta_mediator_effect':
            try:
                mean_value = float(trace.posterior[var].mean().values)
                print(f"Mediator effect mean: {mean_value}")
                summary_data.append({
                    'Variable': 'Mediator Effect',
                    'Model': 'Main', 
                    'Coefficient': mean_value,
                    'AbsCoefficient': abs(mean_value)
                })
            except Exception as e:
                print(f"Error processing beta_mediator_effect: {e}")
            continue
        if var in trace.posterior and hasattr(trace.posterior[var], 'dims') and len(trace.posterior[var].dims) > 1:
            dim_name = var.replace('beta_', '').replace('_coeffs', '')
            if var.startswith('mediator_'):
                dim_name = var.replace('mediator_beta_', '').replace('_coeffs', '')
                if dim_name.startswith('mediator_'):
                    dim_name = dim_name
                else:
                    dim_name = 'mediator_' + dim_name
            
            if dim_name in trace.posterior.coords:
                feature_names = trace.posterior.coords[dim_name].values
                for i, feature in enumerate(feature_names):
                    if dim_name in trace.posterior[var].dims:
                        mean_value = trace.posterior[var].sel({dim_name: feature}).mean().values
                    else:
                        mean_value = trace.posterior[var].values[:, :, i].mean()
                    
                    model_type = "Mediator" if var.startswith('mediator_') else "Main"
                    display_name = f"{feature}"
                    
                    summary_data.append({
                        'Variable': display_name,
                        'Model': model_type,
                        'Coefficient': float(mean_value),
                        'AbsCoefficient': abs(float(mean_value))
                    })
        else:
            try:
                mean_value = float(trace.posterior[var].mean().values)
                model_type = "Mediator" if var.startswith('mediator_') else "Main"
                display_name = var.replace('beta_', '').replace('mediator_', '')
                
                summary_data.append({
                    'Variable': display_name,
                    'Model': model_type, 
                    'Coefficient': mean_value,
                    'AbsCoefficient': abs(mean_value)
                })
            except Exception as e:
                print(f"Error processing variable {var}: {e}")
    
    coeffs_df = pd.DataFrame(summary_data)
    print(f"Coeffs DataFrame:\n{coeffs_df}")
    
    normalized_dfs = []
    for model in coeffs_df['Model'].unique():
        model_df = coeffs_df[coeffs_df['Model'] == model].copy()
        total_abs = model_df['AbsCoefficient'].sum()
        
        model_df['Normalized'] = (model_df['AbsCoefficient'] / total_abs * 100 * 
                                 model_df['Coefficient'].apply(lambda x: 1 if x >= 0 else -1))
        normalized_dfs.append(model_df)
    final_df = pd.concat(normalized_dfs)
    final_df = final_df.sort_values(['Model', 'Normalized'], ascending=[True, True])
    print(f"Total coefficients: {len(final_df)}")
    print(f"Unique variables: {final_df['Variable'].nunique()}")
    
    fig = px.bar(
        final_df,
        x='Normalized',
        y='Variable',
        color='Model',
        orientation='h',
        text='Normalized',
        barmode='group',
        height=figsize[1]*50,
        width=figsize[0]*50,
        labels={'Normalized': 'Relative Importance (%)'},
        title='Normalized Coefficients for Main and Mediator Models'
    )
    
    fig.update_layout(
        xaxis_title="Relative Importance (%)",
        yaxis_title="Features",
        legend_title="Model",
        font=dict(size=12)
    )
    
    fig.update_traces(
        texttemplate='%{x:.2f}%',
        textposition='outside'
    )
    
    return fig


def get_coefficient_table(trace, include_mediator=True):
    """
    Create a summary table of coefficients for both the main and mediator models.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        Trace object containing posterior samples
    include_mediator : bool, default=True
        Whether to include mediator model coefficients
    
    Returns:
    --------
    pandas.DataFrame
        Formatted table of coefficients
    """
    print("All available variables:", list(trace.posterior.data_vars))
    
    main_vars = [var for var in trace.posterior.data_vars 
                if var.startswith('beta_') and 'fourier' not in var and 'offset' not in var]
    
    # Ensure beta_mediator_effect is included
    if 'beta_mediator_effect' not in main_vars and 'beta_mediator_effect' in trace.posterior.data_vars:
        main_vars.append('beta_mediator_effect')
    
    mediator_vars = []
    if include_mediator:
        mediator_vars = [var for var in trace.posterior.data_vars 
                        if var.startswith('mediator_beta_') and 'fourier' not in var and 'offset' not in var]
    
    all_vars = main_vars + mediator_vars
    
    results = []
    
    for var in all_vars:
        if var == 'beta_mediator_effect':
            slice_data = trace.posterior[var].values.flatten()
            hdi_interval = az.hdi(slice_data, hdi_prob=0.94)
            row = {
                'Model': "Main",
                'Variable': var,
                'Feature': "Mediator Effect",
                'Mean': slice_data.mean(),
                'HDI 3%': hdi_interval[0],
                'HDI 97%': hdi_interval[1], 
                'SD': slice_data.std(),
                'P(>0)': (slice_data > 0).mean()
            }
            results.append(row)
        elif var in trace.posterior and hasattr(trace.posterior[var], 'dims') and len(trace.posterior[var].dims) > 1:
            dim_name = var.replace('beta_', '').replace('_coeffs', '')
            if var.startswith('mediator_'):
                dim_name = var.replace('mediator_beta_', '').replace('_coeffs', '')
                if dim_name.startswith('mediator_'):
                    dim_name = dim_name
                else:
                    dim_name = 'mediator_' + dim_name
                
            if dim_name in trace.posterior.coords:
                feature_names = trace.posterior.coords[dim_name].values
                
                for i, feature in enumerate(feature_names):
                    if dim_name in trace.posterior[var].dims:
                        slice_data = trace.posterior[var].sel({dim_name: feature}).values.flatten()
                    else:
                        slice_data = trace.posterior[var].values[:, :, i].flatten()
                    hdi_interval = az.hdi(slice_data, hdi_prob=0.94)
                    display_name = f"{var}[{feature}]"
                    model_type = "Mediator" if var.startswith('mediator_') else "Main"
                    row = {
                        'Model': model_type,
                        'Variable': display_name,
                        'Feature': feature,
                        'Mean': slice_data.mean(),
                        'HDI 3%': hdi_interval[0],
                        'HDI 97%': hdi_interval[1],
                        'SD': slice_data.std(),
                        'P(>0)': (slice_data > 0).mean()
                    }
                    results.append(row)
        else:
            try:
                slice_data = trace.posterior[var].values.flatten()
                hdi_interval = az.hdi(slice_data, hdi_prob=0.94)
                model_type = "Mediator" if var.startswith('mediator_') else "Main"
                
                row = {
                    'Model': model_type,
                    'Variable': var,
                    'Feature': var,
                    'Mean': slice_data.mean(),
                    'HDI 3%': hdi_interval[0],
                    'HDI 97%': hdi_interval[1],
                    'SD': slice_data.std(),
                    'P(>0)': (slice_data > 0).mean()
                }
                results.append(row)
            except Exception as e:
                print(f"Error processing variable {var}: {e}")
                if 'summarize_variable' in globals():
                    summary = summarize_variable(trace, var)
                    model_type = "Mediator" if var.startswith('mediator_') else "Main"
                    
                    row = {
                        'Model': model_type,
                        'Variable': var,
                        'Feature': var,
                        'Mean': summary['mean'].values[0],
                        'HDI 3%': summary['hdi_3%'].values[0],
                        'HDI 97%': summary['hdi_97%'].values[0],
                        'SD': summary['sd'].values[0],
                        'P(>0)': summary['positive_prob'].values[0]
                    }
                    results.append(row)
    
    df = pd.DataFrame(results)
    
    numeric_cols = ['Mean', 'HDI 3%', 'HDI 97%', 'SD', 'P(>0)']
    df[numeric_cols] = df[numeric_cols].applymap(lambda x: round(x, 3) if pd.notnull(x) else x)
    
    df = df.sort_values(['Model', 'Variable'])
    
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    return df


def compare_coefficients(trace):
    """
    Create a side-by-side comparison of main model vs mediator model coefficients.
    Focuses on matched coefficients (like paid media, organic media, etc.)
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        Trace object containing posterior samples
    
    Returns:
    --------
    pandas.DataFrame
        Formatted comparison table
    """
    all_coeffs = get_coefficient_table(trace, include_mediator=True)
    
    main_coeffs = all_coeffs[all_coeffs['Model'] == 'Main'].copy()
    mediator_coeffs = all_coeffs[all_coeffs['Model'] == 'Mediator'].copy()
    mediator_effect = all_coeffs[all_coeffs['Variable'] == 'beta_mediator_effect']
    
    if not mediator_effect.empty:
        print("\n=== Mediator Effect on Main Outcome ===")
        mediator_effect_formatted = mediator_effect[['Variable', 'Mean', 'HDI 3%', 'HDI 97%', 'P(>0)']].copy()
        print(tabulate(mediator_effect_formatted, headers='keys', tablefmt='psql', showindex=False))
    print("\n=== Main Model vs Mediator Model Coefficients ===")
    
    main_coeffs['Type'] = main_coeffs['Variable'].apply(
        lambda x: x.split('_')[1] if 'beta_' in x and '_coeffs' in x else 'other')
    
    mediator_coeffs['Type'] = mediator_coeffs['Variable'].apply(
        lambda x: x.split('_')[2] if 'mediator_beta_' in x and '_coeffs' in x else 'other')
    
    comparison_rows = []
    feature_types = set(main_coeffs['Type']).union(set(mediator_coeffs['Type']))
    feature_types = [ft for ft in feature_types if ft != 'other']
    
    for ftype in feature_types:
        main_type = main_coeffs[main_coeffs['Type'] == ftype]
        mediator_type = mediator_coeffs[mediator_coeffs['Type'] == ftype]
        features = set(main_type['Feature']).union(set(mediator_type['Feature']))
        
        for feature in features:
            main_row = main_type[main_type['Feature'] == feature]
            mediator_row = mediator_type[mediator_type['Feature'] == feature]

            row = {
                'Type': ftype.capitalize(),
                'Feature': feature,
                'Main Mean': main_row['Mean'].values[0] if not main_row.empty else None,
                'Main HDI': f"[{main_row['HDI 3%'].values[0]:.3f}, {main_row['HDI 97%'].values[0]:.3f}]" if not main_row.empty else None,
                'Main P(>0)': main_row['P(>0)'].values[0] if not main_row.empty else None,
                'Mediator Mean': mediator_row['Mean'].values[0] if not mediator_row.empty else None,
                'Mediator HDI': f"[{mediator_row['HDI 3%'].values[0]:.3f}, {mediator_row['HDI 97%'].values[0]:.3f}]" if not mediator_row.empty else None,
                'Mediator P(>0)': mediator_row['P(>0)'].values[0] if not mediator_row.empty else None
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
    mediator_y_train=None, mediator_y_test=None, 
    mediator_train_preds=None, mediator_test_preds=None,
    main_y_label="Watchtime",
    mediator_y_label="Mediator",
    show_plots=True,
    save_path=None
):
    """
    Plot time series of predictions vs actuals for both the main model and mediator model.
    
    Parameters:
    -----------
    train_index, test_index : array-like
        Date indices for training and test sets
    main_y_train, main_y_test : array-like
        Actual values for the main model
    main_train_preds, main_test_preds : array-like
        Predicted values for the main model
    mediator_y_train, mediator_y_test : array-like, optional
        Actual values for the mediator model
    mediator_train_preds, mediator_test_preds : array-like, optional
        Predicted values for the mediator model
    main_y_label, mediator_y_label : str
        Y-axis labels for the respective models
    show_plots : bool
        Whether to display the plots
    save_path : str, optional
        Path to save the figures
    """
    figures = {}
    
    f_main_train = plt.figure(figsize=(12, 8))
    plt.title(f"Main Model | Training Set ({main_y_label})")
    plt.plot(train_index, main_y_train, label="Actual", marker='o')
    plt.plot(train_index, main_train_preds, label="Predicted", marker='x')
    plt.xlabel("Date")
    plt.ylabel(main_y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    figures['main_train'] = f_main_train
    
    f_main_test = plt.figure(figsize=(12, 8))
    plt.title(f"Main Model | Test Set ({main_y_label})")
    plt.plot(test_index, main_y_test, label="Actual", marker='o')
    plt.plot(test_index, main_test_preds, label="Predicted", marker='x')
    plt.xlabel("Date")
    plt.ylabel(main_y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    figures['main_test'] = f_main_test
    
    if (mediator_y_train is not None and mediator_y_test is not None and
        mediator_train_preds is not None and mediator_test_preds is not None):
        
        f_mediator_train = plt.figure(figsize=(12, 8))
        plt.title(f"Mediator Model | Training Set ({mediator_y_label})")
        plt.plot(train_index, mediator_y_train, label="Actual", marker='o')
        plt.plot(train_index, mediator_train_preds, label="Predicted", marker='x')
        plt.xlabel("Date")
        plt.ylabel(mediator_y_label)
        plt.legend()
        plt.grid(True, alpha=0.3)
        figures['mediator_train'] = f_mediator_train
        
        f_mediator_test = plt.figure(figsize=(12, 8))
        plt.title(f"Mediator Model | Test Set ({mediator_y_label})")
        plt.plot(test_index, mediator_y_test, label="Actual", marker='o')
        plt.plot(test_index, mediator_test_preds, label="Predicted", marker='x')
        plt.xlabel("Date")
        plt.ylabel(mediator_y_label)
        plt.legend()
        plt.grid(True, alpha=0.3)
        figures['mediator_test'] = f_mediator_test
    
    if save_path:
        for name, fig in figures.items():
            fig.savefig(f"{save_path}_{name}.png", dpi=300, bbox_inches='tight')
    
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