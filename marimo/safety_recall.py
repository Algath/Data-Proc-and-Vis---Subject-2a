import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import cars
    from sklearn.preprocessing import StandardScaler

    return StandardScaler, pd, plt


@app.cell
def _(pd):
    from pathlib import Path

    # Résout le chemin relativement au notebook (et non au CWD), pour que
    # ça marche aussi bien depuis VS Code que depuis marimo/localhost.
    # On part du dossier du fichier et on remonte jusqu'à trouver "data/".
    def _find_data_dir(start: Path) -> Path:
        for parent in [start, *start.parents]:
            candidate = parent / "data"
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError("Dossier 'data/' introuvable depuis " + str(start))

    try:
        _base = Path(__file__).resolve().parent
    except NameError:
        # __file__ peut être absent selon le mode d'exécution : on retombe sur le CWD.
        _base = Path.cwd()

    _data_dir = _find_data_dir(_base)
    df = pd.read_csv(_data_dir / "nhtsa_vehicle_safety_recall_intelligence_ultimate.csv")

    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pre-processing
    """)
    return


@app.cell
def _(df):
    recall = df['recall_type_label'].unique()
    recall
    return


@app.cell
def _(df):
    df_1 = df[df['recall_type_label'].isin(['Vehicle'])]
    df_1.head()
    return (df_1,)


@app.cell
def _(df_1):
    _components = df_1['component_category'].unique()
    _components
    return


@app.cell
def _(df_1):
    df_2 = df_1[~df_1['component_category'].isin(['Child Seat', 'Other'])]
    df_2.head()
    return (df_2,)


@app.cell
def _(df_2):
    manufacturer_campaign_ids = df_2['manufacturer_campaign_id'].unique()
    vehicle_year = df_2['vehicle_year'].unique()
    vehicle_year_num = df_2['vehicle_year_num'].unique()
    vehicle_era = df_2['vehicle_era'].unique()
    print(manufacturer_campaign_ids)
    print(vehicle_year)
    print(vehicle_year_num)
    print(vehicle_era)
    df_3 = df_2.dropna()
    df_3 = df_3[(df_3['vehicle_year_num'] >= 2014) & (df_3['vehicle_year_num'] <= 2024)]
    return (df_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analyse
    """)
    return


@app.cell
def _(df_3, plt):
    severity_colors = {'CRITICAL': '#d32f2f', 'HIGH': '#f57c00', 'MEDIUM': '#fbc02d', 'LOW': '#388e3c'}
    defect_severity_by_component = df_3.groupby('component_category')['defect_severity'].value_counts().unstack().fillna(0)
    _colors = [severity_colors[col] for col in defect_severity_by_component.columns]
    defect_severity_by_component.plot(kind='bar', stacked=True, color=_colors)
    plt.title('Defect Severity by Component Category')
    plt.xlabel('Component Category')
    plt.ylabel('Count')
    plt.legend(title='Defect Severity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_3):
    defect_severity = df_3['defect_severity'].unique()
    defect_severity
    return (defect_severity,)


@app.cell
def _(df_3, plt):
    defect_severity_by_component_years = df_3.groupby(['vehicle_year_num', 'component_category'])['defect_severity'].value_counts().unstack().fillna(0)
    severity_colors_1 = {'CRITICAL': '#d32f2f', 'HIGH': '#f57c00', 'MEDIUM': '#fbc02d', 'LOW': '#388e3c'}
    _components = defect_severity_by_component_years.index.get_level_values(1).unique()
    n_rows = (len(_components) + 1) // 2
    _fig, _ax = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    _ax = _ax.flatten()
    for i, component in enumerate(_components):
        data = defect_severity_by_component_years.xs(component, level=1)
        _colors = [severity_colors_1.get(col, '#90a4ae') for col in data.columns]
        data.plot(kind='bar', stacked=True, ax=_ax[i], color=_colors)
        _ax[i].set_title(f'Defect Severity for {component}')
        _ax[i].set_xlabel('Vehicle Year')
        _ax[i].set_ylabel('Count')
        _ax[i].legend(title='Defect Severity')
    for j in range(len(_components), len(_ax)):
        _ax[j].set_visible(False)
    plt.tight_layout()
    plt.show()
    return (severity_colors_1,)


@app.cell
def _(df_3, plt, severity_colors_1):
    severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    defects_by_year = df_3.groupby(['vehicle_year_num', 'defect_severity']).size().unstack(fill_value=0).reindex(columns=severity_order)
    _colors = [severity_colors_1[col] for col in defects_by_year.columns]
    _fig, _ax = plt.subplots(figsize=(12, 6))
    defects_by_year.plot(kind='bar', stacked=True, color=_colors, ax=_ax)
    _ax.set_title('Defects by Year')
    _ax.set_xlabel('Vehicle Year')
    _ax.set_ylabel('Count')
    _ax.legend(title='Defect Severity')
    _ax.set_xticklabels([int(y) for y in defects_by_year.index], rotation=45)
    plt.tight_layout()
    plt.show()
    return (defects_by_year,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Approach: going more into hybrid or electrical cars + cars fires
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Normalisation
    """)
    return


@app.cell
def _(
    StandardScaler,
    defect_severity,
    defects_by_year,
    pd,
    plt,
    severity_colors_1,
):
    scaler = StandardScaler()
    standardized_defect_by_year = scaler.fit_transform(defects_by_year)
    standardized_defect_by_year = pd.DataFrame(standardized_defect_by_year, columns=defect_severity)

    _colors = [severity_colors_1[col] for col in standardized_defect_by_year.columns]

    _fig1, _ax1 = plt.subplots(figsize=(12, 6))
    standardized_defect_by_year.plot(kind='bar', stacked=False, color=_colors, ax=_ax1)
    _ax1.set_title('Defect by Year Normalized')
    _ax1.set_xlabel('Vehicle Year')
    _ax1.set_ylabel('Count')
    _ax1.legend(title='Defect Severity')
    _ax1.set_xticklabels([int(y) for y in defects_by_year.index], rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(defect_severity, defects_by_year, pd, plt, severity_colors_1):
    from sklearn.preprocessing import MinMaxScaler

    scaler2 = MinMaxScaler()
    standardized_defect_by_year2 = scaler2.fit_transform(defects_by_year)
    standardized_defect_by_year2 = pd.DataFrame(
        standardized_defect_by_year2,
        columns=defect_severity,
        index=defects_by_year.index  # ← important pour garder les années en index
    )
    _colors = [severity_colors_1[col] for col in standardized_defect_by_year2.columns]
    _fig3, _ax3 = plt.subplots(figsize=(12, 6))
    standardized_defect_by_year2.plot(kind='bar', stacked=False, color=_colors, ax=_ax3)
    _ax3.set_title('Defect by Year Normalized')
    _ax3.set_xlabel('Vehicle Year')
    _ax3.set_ylabel('Count')
    _ax3.legend(title='Defect Severity')
    _ax3.set_xticklabels([int(y) for y in defects_by_year.index], rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt, severity_colors_1):
    severity_counts = (
        df.groupby(['vehicle_year_num', 'defect_severity'])
        .size()
        .unstack(fill_value=0)
        .rename(columns=str.upper)
    )

    # ✅ Filtre sur l'index
    severity_counts = severity_counts[(severity_counts.index >= 2014) & (severity_counts.index <= 2024)]

    for _col in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        if _col not in severity_counts.columns:
            severity_counts[_col] = 0
    severity_counts = severity_counts[['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']]

    vehicle_per_year = (
        df.groupby('vehicle_year_num')['units_affected_num']
        .sum()
        .rename('nb_vehicles')
    )

    result = severity_counts.join(vehicle_per_year)

    # ✅ Colonnes séparées pour le taux
    for _col in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        result[f'{_col}_per_1000'] = (result[_col] / result['nb_vehicles']) * 1000

    result = result.reset_index()

    plot_data = result.set_index('vehicle_year_num')[['CRITICAL_per_1000', 'HIGH_per_1000', 'MEDIUM_per_1000', 'LOW_per_1000']]
    _colors = [severity_colors_1[_col] for _col in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']]

    _fig2, _ax2 = plt.subplots(figsize=(12, 6))
    plot_data.plot(kind='bar', stacked=True, color=_colors, ax=_ax2)
    _ax2.set_title('Defect by Year Normalized')
    _ax2.set_xlabel('Vehicle Year')
    _ax2.set_ylabel('Defects per 1000 vehicles')
    _ax2.legend(title='Defect Severity', labels=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'])
    _ax2.set_xticklabels([int(y) for y in plot_data.index], rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rappel
    """)
    return


@app.cell
def _(df_3, pd):
    df_3['recall_date_parsed'] = pd.to_datetime(df_3['recall_date_parsed'], format='%Y-%m-%d')
    df_3['notification_date_parsed'] = pd.to_datetime(df_3['notification_date_parsed'], format='%Y-%m-%d')

    df_3['jours_ecoules'] = (df_3['notification_date_parsed'] - df_3['recall_date_parsed']).dt.days.clip(lower=0)

    df_3['jours_ecoules']
    return


@app.cell
def _(df_3):
    df_3['jours_ecoules'] = (df_3['notification_date_parsed'] - df_3['recall_date_parsed']).dt.days

    df_3['check'] = df_3['jours_ecoules'] - df_3['days_to_owner_notification']
    df_3[['jours_ecoules', 'days_to_owner_notification', 'check']].describe()
    return


@app.cell
def _(df_3):
    df_3[df_3['check'] < 0][['recall_date_parsed', 'notification_date_parsed', 'jours_ecoules', 'days_to_owner_notification', 'check']]
    return


if __name__ == "__main__":
    app.run()
