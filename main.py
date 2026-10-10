"""
Ingeniería de características con transformadores personalizados.

Flujo:
1. Cargar el dataset de resultados de Colapinto.
2. Diagnóstico de nulos.
3. Dividir en train y test.
4. fit_transform en train y solo transform en test.
5. Comprobar que no hay data leakage.
6. Mostrar que el mismo flujo funciona dentro de un Pipeline.
"""

import os

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from custom_transformers import FeaturesDeCarrera, FeaturesDeFecha, PromedioPorCategoria

RUTA_CSV = "dataset.csv"
URL_API = "http://api.jolpi.ca/ergast/f1/drivers/colapinto/results.json"
# Fecha fija para que antiguedad_en_dias sea reproducible
FECHA_REFERENCIA = "2026-10-10"


def descargar_de_api():
    """Misma extracción que el pipeline ETL anterior: Jolpica-F1, sin autenticación."""
    response = requests.get(URL_API, params={"limit": 100}, timeout=10)
    response.raise_for_status()
    carreras = response.json()["MRData"]["RaceTable"]["Races"]

    filas = []
    for carrera in carreras:
        resultado = carrera["Results"][0]
        filas.append({
            "season": carrera["season"],
            "round": carrera["round"],
            "race_name": carrera["raceName"],
            "date": carrera["date"],
            "constructor": resultado["Constructor"]["name"],
            "grid": resultado["grid"],
            "position": resultado.get("position"),
            "points": resultado["points"],
            "status": resultado["status"],
        })
    df = pd.DataFrame(filas)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["season", "round", "grid", "position", "points"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def generar_datos_simulados(n=41, semilla=0):
    """
    Dataset SIMULADO con las mismas columnas que el real. Solo se usa si no hay
    dataset.csv ni conexión a la API, para que el script corra igual en cualquier
    máquina. Los valores no son resultados reales.
    """
    rng = np.random.default_rng(semilla)
    puntos_por_posicion = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

    grid = rng.integers(8, 20, size=n)
    position = np.clip(grid + rng.integers(-5, 6, size=n), 1, 20).astype(float)
    abandono = rng.random(n) < 0.15
    position[abandono] = np.nan  # quien abandona no tiene posición final

    puntos = [puntos_por_posicion[int(p) - 1] if p <= 10 else 0 for p in np.nan_to_num(position, nan=99)]

    return pd.DataFrame({
        "season": np.where(np.arange(n) < 9, 2024, 2025),
        "round": np.arange(n) + 1,
        "race_name": [f"Gran Premio {i + 1}" for i in range(n)],
        "date": pd.date_range("2024-09-01", periods=n, freq="14D"),
        "constructor": np.where(np.arange(n) < 9, "Williams", "Alpine F1 Team"),
        "grid": grid,
        "position": position,
        "points": puntos,
        "status": np.where(abandono, "Retired", "Finished"),
    })


def cargar_datos():
    """Orden de prioridad: dataset.csv local, API de Jolpica, datos simulados."""
    if os.path.exists(RUTA_CSV):
        print(f"Datos cargados desde {RUTA_CSV}")
        return pd.read_csv(RUTA_CSV, parse_dates=["date"])
    try:
        df = descargar_de_api()
        print("Datos descargados de la API de Jolpica-F1")
        return df
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"No se pudo usar la API ({e}). Se usan datos SIMULADOS.")
        return generar_datos_simulados()


def diagnosticar_nulos(df):
    """
    Conteo de nulos por columna antes de crear variables.
    En pandas, NaN y None se cuentan igual con isna(): None es el nulo de
    Python (tipo object) y NaN es el nulo float de numpy. En las columnas
    numéricas pandas convierte None en NaN automáticamente; en las de texto
    puede quedar como None. Por eso se usa isna() y no == None.
    """
    print("\nNulos por columna:")
    print(df.isna().sum().to_string())


def main():
    df = cargar_datos()
    print(f"\nFilas: {len(df)} | Columnas: {list(df.columns)}")
    diagnosticar_nulos(df)

    # Se divide ANTES de ajustar cualquier transformador
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(df_train)} filas | Test: {len(df_test)} filas")

    fecha = FeaturesDeFecha(fecha_referencia=FECHA_REFERENCIA)
    carrera = FeaturesDeCarrera()
    promedio = PromedioPorCategoria()

    # Train: fit_transform. Test: SOLO transform
    X_train = fecha.fit_transform(df_train)
    X_train = carrera.fit_transform(X_train)
    X_train = promedio.fit_transform(X_train)

    X_test = fecha.transform(df_test)
    X_test = carrera.transform(X_test)
    X_test = promedio.transform(X_test)

    nuevas = ["antiguedad_en_dias", "mes_carrera", "posiciones_ganadas",
              "termino_en_puntos", "no_clasifico", "promedio_puntos_escuderia"]
    print("\nPrimeras filas de train con las variables nuevas:")
    print(X_train[["date", "constructor", "grid", "position"] + nuevas].head().to_string())

    # --- Chequeo de data leakage ---
    print("\nPromedios aprendidos SOLO con train:", {k: round(v, 2) for k, v in promedio.promedios_.items()})
    assert set(promedio.promedios_) == set(df_train["constructor"]), "aprendió categorías que no son de train"
    assert np.isclose(promedio.media_global_, df_train["points"].mean()), "la media global no es la de train"
    assert not X_train["promedio_puntos_escuderia"].isna().any()
    assert not X_test["promedio_puntos_escuderia"].isna().any()
    assert len(df.columns) == len(df_train.columns), "el DataFrame original no debe modificarse"
    print("Chequeo de leakage OK: los parámetros salen solo de train y los DataFrames originales no cambiaron.")

    # --- El mismo flujo dentro de un Pipeline ---
    pipe = Pipeline([
        ("fecha", FeaturesDeFecha(fecha_referencia=FECHA_REFERENCIA)),
        ("carrera", FeaturesDeCarrera()),
        ("promedio", PromedioPorCategoria()),
    ])
    X_train_pipe = pipe.fit_transform(df_train)
    X_test_pipe = pipe.transform(df_test)
    pd.testing.assert_frame_equal(X_train, X_train_pipe)
    pd.testing.assert_frame_equal(X_test, X_test_pipe)
    print("El Pipeline da exactamente el mismo resultado que aplicar los pasos a mano.")


if __name__ == "__main__":
    main()
