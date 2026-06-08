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
    recalls = pd.read_csv(data_dir / "Recalls_Data.csv")

    # On ne garde que les rappels de type "Vehicle" (pas Tire / Equipment / Child Seat).
    recalls = recalls.loc[recalls["Recall Type"].str.strip().eq("Vehicle")].copy()

    # Année du rappel = année de réception du rapport.
    recalls["year"] = pd.to_datetime(
        recalls["Report Received Date"], errors="coerce"
    ).dt.year

    recalls.head()
    return (recalls,)


@app.cell
def _(plt, recalls):
    # Nombre de campagnes de rappel par an (fenêtre 2010-2024 : années complètes).
    rappels_par_an = (
        recalls.loc[(recalls["year"] >= 2010) & (recalls["year"] <= 2024)]
        .groupby("year")
        .size()
        .sort_index()
    )

    fig_rec, ax_rec = plt.subplots(figsize=(10, 5))
    ax_rec.bar(rappels_par_an.index.astype(str), rappels_par_an.values, color="#1f77b4")
    ax_rec.set_title("Recall campaigns per year (vehicles)")
    ax_rec.set_xlabel("Recall year")
    ax_rec.set_ylabel("Number of recall campaigns")
    for _x, _y in zip(rappels_par_an.index.astype(str), rappels_par_an.values):
        ax_rec.text(_x, _y, f"{_y:,}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig_rec.savefig("rappels_par_an.png", dpi=300)
    fig_rec
    return


if __name__ == "__main__":
    app.run()
