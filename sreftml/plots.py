import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.preprocessing as sp
import tensorflow as tf
from scipy.stats import gaussian_kde, linregress

from .utilities import clean_duplicate, n2mfrow, remove_outliers


def hp_search_plot(
    df_grid: pd.DataFrame,
    eval_col: str = "score",
    save_file_path: str | None = None,
) -> plt.Figure:
    """
    Plot the results of hyperparameters search.

    Args:
        df_grid (pd.DataFrame): DataFrame containing the grid of hyperparameters.
        eval_col (str, optional): The column to use for evaluation. Defaults to "score".
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        plt.Figure: The plotted figure.
    """
    df_grid = df_grid.sort_values(eval_col, ascending=False).reset_index(drop=True)
    cols = df_grid.columns.tolist()
    cols.remove(eval_col)
    cols.append(eval_col)
    df_grid_sorted = df_grid[cols]

    string_columns = df_grid_sorted.select_dtypes(include="object").columns
    numeric_columns = df_grid_sorted.select_dtypes(exclude="object").columns

    for i in string_columns:
        df_grid_sorted[i] = sp.LabelEncoder().fit_transform(df_grid_sorted[i])

    scaler = sp.MinMaxScaler()
    df_grid_scaled = pd.DataFrame(
        scaler.fit_transform(df_grid_sorted), columns=df_grid_sorted.columns
    )

    cm = plt.get_cmap("seismic", 2)
    fig = plt.figure(tight_layout=True, dpi=300)
    for i, row in df_grid_scaled.iterrows():
        if i == len(df_grid_sorted) - 1:
            plt.plot(df_grid_scaled.columns, row.values, color=cm(1), lw=4)
        else:
            plt.plot(df_grid_scaled.columns, row.values, color=cm(0))

    for i in string_columns:
        label_unique = df_grid[i].unique()
        scaled_unique = df_grid_scaled[i].unique()
        for label_, scaled_ in zip(label_unique, scaled_unique):
            plt.text(
                i, scaled_, label_, ha="center", va="center", backgroundcolor="white"
            )

    for i in numeric_columns:
        min_val = df_grid_sorted[i].min()
        max_val = df_grid_sorted[i].max()
        plt.text(i, 0, min_val, ha="center", va="center", backgroundcolor="white")
        if min_val != max_val:
            plt.text(i, 1, max_val, ha="center", va="center", backgroundcolor="white")

    for i, val in enumerate(df_grid_scaled.iloc[-1, :]):
        col_name = df_grid_scaled.columns[i]
        if val not in [0, 1] and col_name in numeric_columns:
            plt.text(
                col_name,
                val,
                df_grid_sorted.iloc[-1, i],
                ha="center",
                va="center",
                backgroundcolor="white",
            )

    plt.xticks(rotation=45)
    plt.ylabel("Min-Max Normalized value")

    if save_file_path is not None:
        plt.savefig(save_file_path, transparent=True)

    return fig


def prediction_plot(
    sreft: tf.keras.Model,
    df: pd.DataFrame,
    name_biomarkers: list[str],
    name_biomarkers_display: list[str],
    name_covariates: list[str],
    scaler_y: sp.StandardScaler,
    scaler_cov: sp.StandardScaler,
    biomarkers_is_reversed: dict[str, bool] | None = None,
    biomarkers_to_remove_outlier: list[str] | None = None,
    res: int = 100,
    density: bool = False,
    useOffsetT: bool = True,
    ncol_max: int = 4,
    save_file_path: str | None = None,
    title_size: int = 20,
    label_size: int = 15,
    tick_size: int = 15,
) -> plt.Figure:
    """
    Plot the predictions of the SReFT model.

    Args:
        sreft (tf.keras.Model): The SReFT model.
        df (pd.DataFrame): DataFrame with the data.
        name_biomarkers (list[str]): The names of the biomarkers.
        name_covariates (list[str]): The names of the covariates.
        scaler_y (sp.StandardScaler): The scaler for the y values.
        scaler_cov (sp.StandardScaler): The scaler for the covariate values.
        biomarkers_to_remove_outlier (list[str] | None, optional): The names of the biomarkers to remove outliers. Defaults to None.
        biomarkers_is_reversed (dict[str, bool] | None, optional): Whether the biomarkers are reversed or not. Defaults to None.
        res (int, optional): Resolution of the plot. Defaults to 100.
        density (bool, optional): Whether to plot density or not. Defaults to False.
        useOffsetT (bool, optional): Whether to use offsetT or not. Defaults to True.
        ncol_max (int, optional): Maximum number of columns for subplots. Defaults to 4.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        plt.Figure: The plotted figure.
    """
    n_biomarker = len(name_biomarkers)
    n_covariate = len(name_covariates)
    n_row, n_col = n2mfrow(n_biomarker, ncol_max)
    cm = plt.colormaps["Set1"]
    if biomarkers_to_remove_outlier is None:
        biomarkers_to_remove_outlier = []

    y_data = df[name_biomarkers].values

    if useOffsetT:
        x_data = df.TIME.values + df.offsetT.values
        cov_dummy = np.array([i for i in itertools.product([0, 1], repeat=n_covariate)])
        cov_dummy = np.repeat(cov_dummy, res, axis=0)
        cov_dummy_scaled = scaler_cov.transform(cov_dummy)
        x_model = np.linspace(x_data.min(), x_data.max(), res)
        x_model = np.tile(x_model, 2**n_covariate).reshape(-1, 1)
        x_model = np.concatenate((x_model, cov_dummy_scaled), axis=1)
        y_model = scaler_y.inverse_transform(sreft.model_y(x_model))
        if biomarkers_is_reversed is not None:
            for k, biomarker in enumerate(name_biomarkers):
                if biomarkers_is_reversed[biomarker]:
                    y_model[:, k] = -y_model[:, k]
    else:
        x_data = df.TIME.values

    fig, axs = plt.subplots(
        n_row,
        n_col,
        figsize=(n_col * 3, n_row * 3),
        tight_layout=True,
        dpi=300,
        sharex="row",
    )
    for k, ax in enumerate(axs.flat):
        if k >= n_biomarker:
            ax.axis("off")
            continue

        x_data_tmp = x_data
        y_data_tmp = y_data

        if name_biomarkers[k] in biomarkers_to_remove_outlier:
            outlier_mask = remove_outliers(y_data_tmp[:, k])
            x_data_tmp = x_data_tmp[outlier_mask]
            y_data_tmp = y_data_tmp[outlier_mask]
        if biomarkers_is_reversed is not None:
            if biomarkers_is_reversed[name_biomarkers[k]]:
                y_data_tmp[:, k] = -y_data_tmp[:, k]

        if density:
            x_ = x_data_tmp[~np.isnan(y_data_tmp[:, k])]
            y_ = y_data_tmp[~np.isnan(y_data_tmp[:, k]), k]
            if np.var(x_) == 0:
                z = gaussian_kde(y_)(y_)
            else:
                xy = np.vstack([x_, y_])
                z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            ax.scatter(x_[idx], y_[idx], c=z[idx], s=2, label="_nolegend_")
        else:
            ax.scatter(
                x_data_tmp, y_data_tmp[:, k], c="silver", s=2, label="_nolegend_"
            )

        if useOffsetT:
            for i in range(2**n_covariate):
                ax.plot(
                    x_model[res * i : res * (i + 1), 0],
                    y_model[res * i : res * (i + 1), k],
                    c=cm(i),
                    lw=4,
                )
                ax.minorticks_on()
                ax.tick_params(axis="x", labelsize=tick_size)
                ax.tick_params(axis="y", labelsize=tick_size)
                ax.tick_params(axis="x", which="minor", length=7)
                ax.tick_params(axis="y", which="minor", length=7)
                # ax.set_xlabel("Disease Time (year)", fontsize=label_size, fontweight="bold")
                ax.set_xlabel("")
        else:
            ax.set_xlabel(
                "Observation Period (year)", fontsize=label_size, fontweight="bold"
            )

        ax.set_title(name_biomarkers_display[k], fontsize=title_size, fontweight="bold")

    fig.supxlabel("疾患時間（年）", fontsize=label_size, fontweight="heavy")

    if n_covariate > 0:
        legend_labels = [
            ", ".join(format(i, f"0{n_covariate}b")) for i in range(2**n_covariate)
        ]
        fig.legend(
            loc="center",
            framealpha=0,
            bbox_to_anchor=(1.1, 0.5),
            ncol=1,
            title=", ".join(name_covariates),
            labels=legend_labels,
        )

    if save_file_path is not None:
        fig.savefig(save_file_path, transparent=True, bbox_inches="tight")

    return fig


def get_regression_line_label(x: pd.Series, y: pd.Series) -> str:
    """
    Generate a label for a line fitted to the given x and y data using linear regression.

    Parameters:
        x (pd.Series): Series of x-axis data.
        y (pd.Series): Series of y-axis data.

    Returns:
        str: Label for the fitted line, including slope, intercept, and R-squared value.

    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    label_line_1 = rf"$y={slope:.3f}x{'' if intercept < 0 else '+'}{intercept:.3f}$"
    label_line_2 = rf"$R^2:{r_value**2:.2f}$"
    label_line = label_line_1 + "\n" + label_line_2

    return label_line


def single_panel_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue: str | None = None,
    duplicate_key: list[str] | str | None = None,
    density: bool = False,
    identity: bool = False,
    save_file_path: str | None = None,
) -> sns.axisgrid.FacetGrid:
    """
    Draw a scatter plot using a single panel.

    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): X-axis column in df.
        y_col (str): Y-axis column in df.
        hue (str | None, optional): Column to stratify the plot. Defaults to None.
        duplicate_key (list[str] | str | None, optional): Specify the column name(s) from which duplicates are to be removed. Defaults to None.
        density (bool, optional): Whether to plot density. Defaults to False.
        identity (bool, optional): Whether to plot identity line. Defaults to False.
        save_file_path (str, optional): The path where the plot will be saved. Default to None.

    Returns:
        sns.axisgrid.FacetGrid: FacetGrid object with the scatter plot.
    """
    if density:
        hue = None
        warnings.warn("Since density is True, the hue option is ignored.")

    if hue:
        df_ = clean_duplicate(df, [x_col, y_col, hue], duplicate_key)
        unique_hues = np.sort(df_[hue].unique())
        line_kws_ = None
    else:
        df_ = clean_duplicate(df, [x_col, y_col], duplicate_key)
        unique_hues = [None]
        line_kws_ = {"color": "red"}

    scatter_kws_ = {"alpha": 0.5, "s": 20, "edgecolor": "none"}
    if density:
        xy = df_[[x_col, y_col]].values.T
        z = gaussian_kde(xy)(xy)
        scatter_kws_.update({"c": z, "color": None, "cmap": "viridis"})

    g = sns.lmplot(
        data=df_,
        x=x_col,
        y=y_col,
        hue=hue,
        hue_order=unique_hues,
        scatter_kws=scatter_kws_,
        line_kws=line_kws_,
    )
    g.figure.set_dpi(300)

    if identity:
        if df[y_col].max() < df[x_col].min() or df[x_col].max() < df[y_col].min():
            warnings.warn(
                f"The data range of {x_col} and {y_col} is not covered, although idenntity=True. Skip drawing of identity line."
            )
        else:
            min_ = df[[x_col, y_col]].min().max()
            max_ = df[[x_col, y_col]].max().min()
            g.axes[0, 0].plot([min_, max_], [min_, max_], "k--")

    if hue:
        g.axes[0, 0].legend(
            ["_nolegend_", "dummy text", "_nolegned_"] * len(unique_hues)
        )
        for idx, h in enumerate(unique_hues):
            df_hue = df_.loc[df_[hue] == h]
            label_line = get_regression_line_label(df_hue[x_col], df_hue[y_col])
            g.axes[0, 0].get_legend().get_texts()[idx].set_text(label_line)
    else:
        label_line = get_regression_line_label(df_[x_col], df_[y_col])
        g.axes[0, 0].legend(labels=["_nolegend_", label_line])

    if save_file_path is not None:
        plt.savefig(save_file_path, transparent=True, dpi=300)

    return g


def plot_jointplot_with_regression(
    df, x_col, y_col, duplicate_key=None, save_file_path=None
):
    """
    Create a joint plot with scatter plot, regression line, and regression line label using cleaned data.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        duplicate_key (list[str] | str | None): Key columns to check for duplicates.
        save_file_path (str | None): Path to save the plot. If None, plot is not saved.

    Returns:
        sns.JointGrid: JointGrid object with the joint plot.
    """
    # Clean the DataFrame
    df_cleaned = clean_duplicate(df, [x_col, y_col], duplicate_key)

    # Ensure there's enough data left after cleaning
    if df_cleaned.empty or len(df_cleaned) < 2:
        warnings.warn("Insufficient data after cleaning.")
        return

    # Create the joint plot
    g = sns.jointplot(
        x=x_col,
        y=y_col,
        data=df_cleaned,
        kind="reg",
        height=8,
        line_kws={"color": "red"},
        marginal_kws={"kde": False},
    )

    # Calculate regression line label
    label = get_regression_line_label(df_cleaned[x_col], df_cleaned[y_col])

    # Add regression line label to the plot
    g.ax_joint.annotate(label, xy=(0.1, 0.9), xycoords="axes fraction", fontsize=20)

    # Set font size for each element
    g.ax_joint.set_xlabel(x_col, fontsize=20)
    g.ax_joint.set_ylabel(y_col, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Save the plot if a file path is provided
    if save_file_path:
        g.savefig(save_file_path, transparent=True, dpi=300)

    return g
