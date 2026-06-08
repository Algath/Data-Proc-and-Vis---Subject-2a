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

    data_dir = _find_data_dir(_base)
    df = pd.read_csv(data_dir / "nhtsa_vehicle_safety_recall_intelligence_ultimate.csv")

    df.head()
    return data_dir, df


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
    # ATTENTION: df_2.dropna() supprimait ~98% des lignes (44984 -> 915) car des
    # colonnes comme MFR_COMP_PTNO / manufacture_date_* sont presque vides.
    # On ne supprime que les lignes nulles sur les colonnes réellement utilisées.
    df_3 = df_2.dropna(subset=['vehicle_year_num', 'defect_severity', 'component_category'])
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
    _totals = defects_by_year.sum(axis=1)
    _ax.bar_label(_ax.containers[-1], labels=[int(t) for t in _totals], label_type='edge', padding=3, fontsize=8)
    _ax.set_title('Defects by Year')
    _ax.set_xlabel('Vehicle Year')
    _ax.set_ylabel('Count')
    _ax.legend(title='Defect Severity')
    _ax.set_xticklabels([int(y) for y in defects_by_year.index], rotation=45)
    plt.tight_layout()
    plt.savefig('plots/defects_by_year.png', dpi=300)
    plt.show()
    return (defects_by_year,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Électronique/logiciel vs mécanique : évolution générale

    On regroupe les composants en deux grandes familles d'ingénierie pour comparer leur
    évolution par millésime :
    - **Électronique / logiciel** : Electrical, Software / EV
    - **Mécanique** : Engine / Powertrain, Fuel System, Brakes, Steering, Suspension, Tires / Wheels

    Les catégories de sécurité passive (Air Bag, Seat Belts, Visibility, Fire Risk) sont
    exclues de *cette* comparaison pour garder deux familles homogènes.

    Pour comparer les *trajectoires* (et pas les niveaux absolus, la mécanique étant ~3x plus
    volumineuse), on indexe chaque famille en **base 100 sur le premier millésime (2014)** :
    on voit ainsi le % d'évolution, et si la courbe a un pic ou reste stable.
    """)
    return


@app.cell
def _(df_3, plt):
    _electronics = ['Electrical', 'Software / EV']
    _mechanical = ['Engine / Powertrain', 'Fuel System', 'Brakes', 'Steering', 'Suspension', 'Tires / Wheels']

    def _famille(c):
        if c in _electronics:
            return 'Electronics / Software'
        if c in _mechanical:
            return 'Mechanical'
        return None  # sécurité passive : exclue de la comparaison

    family_by_year = (
        df_3.assign(famille=df_3['component_category'].map(_famille))
        .dropna(subset=['famille'])
        .groupby(['vehicle_year_num', 'famille']).size()
        .unstack(fill_value=0)
    )

    # Indice base 100 sur le premier millésime (2014) pour comparer les trajectoires.
    family_index = family_by_year / family_by_year.iloc[0] * 100

    _fig, _ax = plt.subplots(figsize=(12, 6))
    family_index['Electronics / Software'].plot(ax=_ax, marker='o', color='#1f77b4')
    family_index['Mechanical'].plot(ax=_ax, marker='o', color='#7f7f7f')
    _ax.axhline(100, color='black', linewidth=0.8, linestyle='--')
    _ax.set_title('Recall trend by component family (model year 2014 = 100)')
    _ax.set_xlabel('Vehicle model year')
    _ax.set_ylabel('Index (2014 = 100)')
    _ax.legend(title='Family')
    plt.tight_layout()
    plt.savefig('plots/recall_trend_by_family.png', dpi=300)
    plt.show()
    return (family_by_year,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Version « présentation » : fenêtre 2014-2022 (coupée au pic)

    En s'arrêtant au pic de 2022 (et en ne montrant pas la retombée 2023-2024 due au manque de
    recul des millésimes récents), les deux courbes deviennent proprement croissantes et l'écart
    électronique/mécanique saute aux yeux. Donnée brute non modifiée : seul le cadrage temporel
    est choisi.
    """)
    return


@app.cell
def _(family_by_year, plt):
    # Fenêtre coupée à 2022 (le pic), réindexée base 100 sur 2014.
    _win = family_by_year[family_by_year.index <= 2022]
    family_index_2022 = _win / _win.iloc[0] * 100

    _fig, _ax = plt.subplots(figsize=(12, 6))
    family_index_2022['Electronics / Software'].plot(ax=_ax, marker='o', color='#1f77b4')
    family_index_2022['Mechanical'].plot(ax=_ax, marker='o', color='#7f7f7f')
    _ax.axhline(100, color='black', linewidth=0.8, linestyle='--')
    _ax.set_title('Recall trend by component family, 2014-2022 (2014 = 100)')
    _ax.set_xlabel('Vehicle model year')
    _ax.set_ylabel('Index (2014 = 100)')
    _ax.legend(title='Family')

    # Valeur de l'indice annotée sur chaque point, pour les deux familles.
    for _col, _va, _dy in [('Electronics / Software', 'bottom', 6), ('Mechanical', 'top', -6)]:
        for _year, _val in family_index_2022[_col].items():
            _ax.annotate(f'{_val:.0f}', xy=(_year, _val), xytext=(0, _dy),
                         textcoords='offset points', ha='center', va=_va, fontsize=8)
    plt.tight_layout()
    plt.savefig('plots/recall_trend_by_family_2014_2022.png', dpi=300)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### La réalité : différence de volume absolu

    L'indice base 100 masque que la mécanique pèse ~3x plus de rappels que l'électronique sur
    chaque millésime. En volume absolu, l'électronique reste une part minoritaire : c'est ce que
    le graphe « présentation » fait oublier.
    """)
    return


@app.cell
def _(family_by_year, plt):
    # Volumes absolus : la mécanique domine largement chaque millésime.
    _fig, _ax = plt.subplots(figsize=(12, 6))
    family_by_year[['Mechanical', 'Electronics / Software']].plot(
        kind='area', stacked=True, ax=_ax, color=['#7f7f7f', '#1f77b4'])
    _ax.set_title('Absolute recall volume by component family and model year')
    _ax.set_xlabel('Vehicle model year')
    _ax.set_ylabel('Number of recalls')
    _ax.legend(title='Family')
    plt.tight_layout()
    plt.savefig('plots/recall_volume_by_family.png', dpi=300)
    plt.show()
    return


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
    totals = plot_data.sum(axis=1)
    _ax2.bar_label(
        _ax2.containers[-1],
        labels=[f'{v:.4f}' for v in totals],
        label_type='edge', padding=3, fontsize=8,
    )
    plt.tight_layout()
    plt.savefig('plots/defect_by_year_normalized_per_1000.png', dpi=300)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Taux normalisé par la flotte (immatriculations)
    """)
    return


@app.cell
def _(data_dir, df, pd, plt):
    # Flotte immatriculée par année calendaire (national = somme de tous les États).
    _reg = pd.read_csv(data_dir / "Motor_Vehicle_Registrations__1900_-_2024__MV-1_.csv")
    fleet_per_year = (
        _reg[_reg['Category'].isin(['Auto', 'Truck'])]
        .groupby('Year')['Vehicles'].sum()
        .rename('fleet')
    )

    # Unité = campagne de rappel UNIQUE. Chaque ligne du dataset est un millésime/modèle
    # d'une campagne (~9 lignes par campagne), donc on déduplique via recall_campaign_id.
    _veh = df[df['recall_type_label'] == 'Vehicle']
    campaigns_per_year = (
        _veh.groupby('recall_year')['recall_campaign_id'].nunique().rename('campaigns')
    )

    # Période commune aux deux sources (immatriculations: ... -> 2024).
    recall_rate = pd.concat([campaigns_per_year, fleet_per_year], axis=1).dropna()
    recall_rate = recall_rate[(recall_rate.index >= 2014) & (recall_rate.index <= 2024)]
    recall_rate['per_million'] = recall_rate['campaigns'] / recall_rate['fleet'] * 1_000_000

    _fig, _ax = plt.subplots(figsize=(12, 6))
    recall_rate['per_million'].plot(kind='bar', color='#4C78A8', ax=_ax)
    _ax.set_title('Recall campaigns per million registered vehicles (by recall year)')
    _ax.set_xlabel('Recall year')
    _ax.set_ylabel('Campaigns / million vehicles')
    _ax.set_xticklabels([int(y) for y in recall_rate.index], rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Over-engineering : montée de l'électronique / du logiciel
    """)
    return


@app.cell
def _(df, plt):
    # Part des rappels par famille de composant et par millésime (composition, en %).
    _v = df[df['recall_type_label'] == 'Vehicle'].dropna(subset=['vehicle_year_num', 'component_category'])
    _v = _v[(_v['vehicle_year_num'] >= 2014) & (_v['vehicle_year_num'] <= 2024)]

    _electronics = ['Electrical', 'Software / EV']
    _mechanical = ['Engine / Powertrain', 'Brakes', 'Steering', 'Suspension', 'Tires / Wheels', 'Fuel System']

    def _grp(c):
        if c in _electronics:
            return 'Electronics / Software'
        if c in _mechanical:
            return 'Mechanical'
        return 'Other'

    _mix = (
        _v.assign(groupe=_v['component_category'].map(_grp))
        .groupby(['vehicle_year_num', 'groupe']).size()
        .unstack(fill_value=0)
    )
    component_share = _mix.div(_mix.sum(axis=1), axis=0) * 100
    component_share = component_share[['Electronics / Software', 'Mechanical', 'Other']]

    _fig, _ax = plt.subplots(figsize=(12, 6))
    component_share.plot(kind='area', stacked=True, ax=_ax,
                         color=['#1f77b4', '#7f7f7f', '#d9d9d9'])
    _ax.set_title("Recall composition by model year: rise of electronics / software")
    _ax.set_xlabel('Vehicle model year')
    _ax.set_ylabel('Share of recalls (%)')
    _ax.set_ylim(0, 100)
    _ax.legend(title='Component family', loc='lower left')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt):
    # Catégorie "Software / EV" : mode de défaillance quasi inexistant avant ~2016.
    _v = df[df['recall_type_label'] == 'Vehicle'].dropna(subset=['vehicle_year_num', 'component_category'])
    _sw = _v[_v['component_category'] == 'Software / EV'].groupby('vehicle_year_num').size()
    _sw = _sw[(_sw.index >= 2014) & (_sw.index <= 2024)]

    _fig, _ax = plt.subplots(figsize=(12, 5))
    _ax.bar([int(y) for y in _sw.index], _sw.values, color='#1f77b4')
    _ax.set_title("« Software / EV » recalls by model year (emerging failure mode)")
    _ax.set_xlabel('Vehicle model year')
    _ax.set_ylabel('Number of recalls')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Version « présentation » (mensonge) : le pic de la part électronique

    On met en avant la **part** des rappels électroniques/logiciels, qui culmine à un record de
    **32 % en 2023**. Cadrage choisi pour appuyer le récit : axe Y resserré (la montée paraît
    spectaculaire) et fenêtre arrêtée au **pic 2023** (on ne montre pas la retombée 2024 ni le
    volume absolu). Donnée brute non modifiée — seule la mise en scène est orientée.
    """)
    return


@app.cell
def _(df, plt):
    # Graphe "mensonge" : part électronique/logiciel par millésime, mise en scène pour
    # dramatiser le pic 2023 (axe Y tronqué + fenêtre coupée au pic).
    _v = df[df['recall_type_label'] == 'Vehicle'].dropna(subset=['vehicle_year_num', 'component_category'])
    _v = _v[(_v['vehicle_year_num'] >= 2014) & (_v['vehicle_year_num'] <= 2023)]

    _elec = ['Electrical', 'Software / EV']
    _v = _v.assign(is_elec=_v['component_category'].isin(_elec))
    _share = _v.groupby('vehicle_year_num')['is_elec'].mean() * 100
    _years = [int(y) for y in _share.index]

    _fig, _ax = plt.subplots(figsize=(12, 6))
    _ax.fill_between(_years, _share.values, _share.min() - 1, color='#1f77b4', alpha=0.15)
    _ax.plot(_years, _share.values, marker='o', color='#1f77b4', linewidth=2)

    # On souligne le pic 2023.
    _peak_y = _years[-1]
    _peak_v = _share.values[-1]
    _ax.scatter([_peak_y], [_peak_v], s=120, color='#d32f2f', zorder=5)
    _ax.annotate(f'{_peak_v:.0f}% — all-time high',
                 xy=(_peak_y, _peak_v), xytext=(_peak_y - 2.2, _peak_v - 2.5),
                 fontsize=12, fontweight='bold', color='#d32f2f',
                 arrowprops=dict(arrowstyle='->', color='#d32f2f'))

    _ax.set_title('Electronics / software defects keep climbing — modern cars are over-engineered')
    _ax.set_xlabel('Vehicle model year')
    _ax.set_ylabel('Share of recalls (%)')
    _ax.set_ylim(_share.min() - 1, _share.max() + 1.5)  # axe tronqué (dramatise la montée)
    _ax.set_xticks(_years)
    _ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/electronics_share_peak.png', dpi=300)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## La réalité derrière le « 32 % » : part (%) vs volume absolu

    Le graphe de composition fait grimper la **part** électronique de 19 % (2014) à 32 % (2023).
    Mais en superposant le **volume absolu** de rappels électroniques, on voit qu'il plafonne
    (pic à 2022) puis recule : la part n'augmente autant que parce que le **dénominateur**
    s'effondre sur les millésimes récents (censurage à droite, données encore incomplètes).
    La hausse de la part est donc un effet de dénominateur, pas une vraie explosion.
    """)
    return


@app.cell
def _(df, plt):
    # Démonstration "% vs volume" : part électronique (métrique mise en avant) confrontée
    # au volume absolu de rappels électroniques (la réalité), par millésime.
    _v = df[df['recall_type_label'] == 'Vehicle'].dropna(subset=['vehicle_year_num', 'component_category'])
    _v = _v[(_v['vehicle_year_num'] >= 2014) & (_v['vehicle_year_num'] <= 2024)]

    _elec = ['Electrical', 'Software / EV']
    _v = _v.assign(is_elec=_v['component_category'].isin(_elec))
    elec_count = _v.groupby('vehicle_year_num')['is_elec'].sum()
    total_count = _v.groupby('vehicle_year_num').size()
    elec_share = elec_count / total_count * 100
    _years = [int(y) for y in elec_share.index]

    _fig, _ax_share = plt.subplots(figsize=(12, 6))
    _ax_vol = _ax_share.twinx()

    # Part (%) : la métrique trompeuse (axe gauche).
    _l1, = _ax_share.plot(_years, elec_share.values, marker='o', color='#1f77b4',
                          label='Share of recalls (%)')
    # Volume absolu : la réalité (axe droit).
    _l2, = _ax_vol.plot(_years, elec_count.values, marker='s', linestyle='--',
                        color='#d62728', label='Absolute recall count')

    # Zone censurée à droite (millésimes récents, données encore incomplètes).
    _ax_share.axvspan(2021.5, 2024.5, color='#000000', alpha=0.06)
    _ax_share.text(2023, _ax_share.get_ylim()[1] * 0.96, 'right-censored\n(incomplete)',
                   ha='center', va='top', fontsize=9, color='#555555')

    _ax_share.set_title('Electronics recalls: rising share, but flat / declining volume')
    _ax_share.set_xlabel('Vehicle model year')
    _ax_share.set_ylabel('Share of recalls (%)', color='#1f77b4')
    _ax_vol.set_ylabel('Absolute recall count', color='#d62728')
    _ax_share.tick_params(axis='y', labelcolor='#1f77b4')
    _ax_vol.tick_params(axis='y', labelcolor='#d62728')
    _ax_vol.set_ylim(bottom=0)  # axe volume honnête : démarre à 0
    _ax_share.set_xticks(_years)
    _ax_share.legend(handles=[_l1, _l2], loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/electronics_share_vs_volume.png', dpi=300)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
