import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from custom_transformers import FeaturesDeCarrera, FeaturesDeFecha, PromedioPorCategoria


@pytest.fixture
def df_ejemplo():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-09-01", "2024-09-15", "2024-10-06", "2024-10-20"]),
        "constructor": ["Williams", "Williams", "Alpine", "Alpine"],
        "grid": [18, 19, 11, 16],
        "position": [12, 8, np.nan, 10],
        "points": [0.0, 4.0, 0.0, 1.0],
    })


def test_fecha_antiguedad_y_mes(df_ejemplo):
    t = FeaturesDeFecha(fecha_referencia="2024-10-20")
    out = t.fit_transform(df_ejemplo)
    assert out["antiguedad_en_dias"].tolist() == [49, 35, 14, 0]
    assert out["mes_carrera"].tolist() == [9, 9, 10, 10]


def test_no_modifica_el_dataframe_original(df_ejemplo):
    columnas_antes = list(df_ejemplo.columns)
    for t in [FeaturesDeFecha(), FeaturesDeCarrera(), PromedioPorCategoria()]:
        t.fit_transform(df_ejemplo)
    assert list(df_ejemplo.columns) == columnas_antes


def test_carrera_posiciones_ganadas_y_nulos(df_ejemplo):
    out = FeaturesDeCarrera().fit_transform(df_ejemplo)
    assert out["posiciones_ganadas"].iloc[0] == 6
    assert out["posiciones_ganadas"].iloc[1] == 11
    assert np.isnan(out["posiciones_ganadas"].iloc[2])  # no clasificó
    assert out["no_clasifico"].tolist() == [0, 0, 1, 0]
    assert out["termino_en_puntos"].tolist() == [0, 1, 0, 1]


def test_carrera_acepta_position_con_tipo_Int64():
    df = pd.DataFrame({"grid": [10, 5], "position": pd.array([3, None], dtype="Int64")})
    out = FeaturesDeCarrera().fit_transform(df)
    assert out["no_clasifico"].tolist() == [0, 1]


def test_promedio_aprende_solo_de_lo_que_ve_en_fit(df_ejemplo):
    train = df_ejemplo.iloc[:2]  # solo Williams
    t = PromedioPorCategoria().fit(train)
    assert t.promedios_ == {"Williams": 2.0}
    assert t.media_global_ == 2.0


def test_promedio_categoria_nueva_usa_media_global_de_train(df_ejemplo):
    train = df_ejemplo.iloc[:2]
    test = df_ejemplo.iloc[2:]  # Alpine, que no estaba en train
    t = PromedioPorCategoria().fit(train)
    out = t.transform(test)
    assert (out["promedio_puntos_escuderia"] == 2.0).all()


def test_promedio_no_cambia_si_cambia_el_test(df_ejemplo):
    """Si hubiera leakage, los parámetros cambiarían al transformar otro test."""
    train = df_ejemplo.iloc[:2]
    t = PromedioPorCategoria().fit(train)
    antes = dict(t.promedios_)
    t.transform(df_ejemplo.iloc[2:])
    assert t.promedios_ == antes


def test_promedio_funciona_con_columna_category(df_ejemplo):
    df = df_ejemplo.copy()
    df["constructor"] = df["constructor"].astype("category")
    out = PromedioPorCategoria().fit(df.iloc[:2]).transform(df)
    assert out["promedio_puntos_escuderia"].notna().all()


def test_transform_sin_fit_da_error(df_ejemplo):
    with pytest.raises(NotFittedError):
        PromedioPorCategoria().transform(df_ejemplo)


def test_columna_faltante_da_error_claro(df_ejemplo):
    with pytest.raises(KeyError):
        FeaturesDeFecha(columna_fecha="no_existe").fit(df_ejemplo)
