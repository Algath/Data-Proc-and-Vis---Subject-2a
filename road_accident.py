import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt

    return pd, plt


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
    df = pd.read_csv(data_dir / "road_accident_data.csv")

    # Année de l'accident (format jour/mois/année).
    df["year"] = pd.to_datetime(
        df["Accident Date"], dayfirst=True, errors="coerce"
    ).dt.year

    # Normalise la sévérité (le dataset contient une coquille "Fetal").
    df["severity"] = df["Accident_Severity"].replace({"Fetal": "Fatal"})

    df.head()
    return (df,)


@app.cell
def _(df, plt):
    # Nombre d'accidents par an.
    accidents_par_an = df.groupby("year").size().sort_index()

    fig_acc, ax_acc = plt.subplots(figsize=(7, 4))
    ax_acc.bar(accidents_par_an.index.astype(str), accidents_par_an.values, color="#1f77b4")
    ax_acc.set_title("Number of accidents per year")
    ax_acc.set_xlabel("Year")
    ax_acc.set_ylabel("Number of accidents")
    for x, y in zip(accidents_par_an.index.astype(str), accidents_par_an.values):
        ax_acc.text(x, y, f"{y:,}", ha="center", va="bottom")
    fig_acc
    return


@app.cell
def _(df, plt):
    # Nombre de morts par an (accidents de sévérité "Fatal").
    morts_par_an = (
        df[df["severity"] == "Fatal"].groupby("year").size().sort_index()
    )

    fig_morts, ax_morts = plt.subplots(figsize=(7, 4))
    ax_morts.bar(morts_par_an.index.astype(str), morts_par_an.values, color="#d62728")
    ax_morts.set_title("Fatal accidents per year")
    ax_morts.set_xlabel("Year")
    ax_morts.set_ylabel("Fatal accidents")
    for xi, yi in zip(morts_par_an.index.astype(str), morts_par_an.values):
        ax_morts.text(xi, yi, f"{yi:,}", ha="center", va="bottom")
    fig_morts
    return


@app.cell
def _(df, plt):
    # Victimes par sévérité et par an (ouverture du récit : le poids humain).
    # On somme les victimes (Number_of_Casualties), réparties selon la sévérité
    # de l'accident, du plus grave au plus léger.
    severity_order = ["Fatal", "Serious", "Slight"]
    severity_colors = {"Fatal": "#d32f2f", "Serious": "#f57c00", "Slight": "#fbc02d"}

    victimes_par_severite = (
        df.groupby(["year", "severity"])["Number_of_Casualties"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=severity_order)
        .sort_index()
    )

    fig_vict, ax_vict = plt.subplots(figsize=(7, 5))
    _bottom = 0
    for _sev in severity_order:
        _vals = victimes_par_severite[_sev].values
        ax_vict.bar(
            victimes_par_severite.index.astype(str),
            _vals,
            bottom=_bottom,
            color=severity_colors[_sev],
            label=_sev,
        )
        _bottom = _bottom + _vals

    # Total annuel au sommet de chaque barre.
    _totaux = victimes_par_severite.sum(axis=1)
    for _x, _tot in zip(victimes_par_severite.index.astype(str), _totaux.values):
        ax_vict.text(_x, _tot, f"{_tot:,}", ha="center", va="bottom", fontweight="bold")

    ax_vict.set_title("Road casualties by severity")
    ax_vict.set_xlabel("Year")
    ax_vict.set_ylabel("Number of casualties")
    ax_vict.legend(title="Accident severity")
    fig_vict.savefig("victimes_par_severite.png", dpi=300)
    fig_vict
    return


@app.cell
def _(df, plt):
    # Chiffre-choc d'ouverture : totaux cumulés sur la période (2021-2022).
    # Accidents = nombre de lignes ; Victimes = somme de Number_of_Casualties.
    total_accidents = len(df)
    total_victimes = int(df["Number_of_Casualties"].sum())

    _labels = ["Accidents", "Casualties"]
    _valeurs = [total_accidents, total_victimes]
    _couleurs = ["#1f77b4", "#d32f2f"]

    fig_tot, ax_tot = plt.subplots(figsize=(6, 5))
    ax_tot.bar(_labels, _valeurs, color=_couleurs)
    for _x, _v in zip(_labels, _valeurs):
        ax_tot.text(_x, _v, f"{_v:,}", ha="center", va="bottom", fontweight="bold")

    ax_tot.set_title("Road toll over the period (2021-2022)")
    ax_tot.set_ylabel("Cumulative total")
    ax_tot.margins(y=0.12)  # marge en haut pour ne pas couper les étiquettes
    fig_tot.savefig("plots/bilan_total.png", dpi=300, bbox_inches="tight")
    fig_tot
    return


if __name__ == "__main__":
    app.run()
