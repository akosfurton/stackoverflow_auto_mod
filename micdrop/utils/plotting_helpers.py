import pandas as pd
from pdpbox import pdp


def plot_subscription_rate_by_var(df, col_nm):
    df_plot = df.copy()

    if col_nm == "platform":
        df_plot = df_plot[df_plot["platform"] != "undefined platform"]

    df_plot["tmp_col"] = df_plot.index
    df_plot_grp = df_plot.groupby(col_nm).agg(
        {"subscriber": "mean", "tmp_col": "count"}
    )

    ax = df_plot_grp.plot.scatter(
        x="tmp_col",
        y="subscriber",
        title=f"Subscription rate by {col_nm}",
        figsize=(12, 5),
    )
    ax.set_xlabel("Segment Size")
    ax.set_ylabel("Click to Subscribe Rate")
    ax.axhline(df_plot["subscriber"].mean(), c="blue", linestyle="--")
    for k, v in df_plot_grp.iterrows():
        ax.annotate(k, (v.values[1], v.values[0]), fontsize=14)


def plot_partial_dependence(clf, df, y_var, col_nm):

    pdp_iso = pdp.pdp_isolate(
        model=clf,
        dataset=df,
        model_features=list(df),
        feature=[x for x in df.columns if col_nm in x],
        num_grid_points=50,
    )

    ax = (
        pd.Series(pdp_iso.pdp, index=pdp_iso.display_columns)
        .sort_values(ascending=False)
        .plot(
            kind="bar",
            title=col_nm,
            ylim=(pdp_iso.pdp.min() * 0.9, pdp_iso.pdp.max() * 1.05),
            figsize=(12, 5),
        )
    )
    ax.axhline(y_var.mean(), c="blue", linestyle="--")
    ax.set_xlabel(col_nm)
    ax.set_ylabel("Average Prediction")
