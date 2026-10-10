"""
Transformadores personalizados para el dataset de resultados de Franco Colapinto.

Los tres siguen la estructura de Scikit-Learn: heredan de BaseEstimator y
TransformerMixin e implementan fit(X, y=None) y transform(X).

- FeaturesDeFecha y FeaturesDeCarrera aplican reglas fijas, así que su fit no
  aprende nada y solo devuelve self.
- PromedioPorCategoria SÍ aprende algo en fit (el promedio de puntos por
  escudería), y por eso es el que hay que cuidar para evitar data leakage:
  se ajusta solo con train y después se aplica a test con transform.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _verificar_columnas(X, columnas):
    """Lanza un error claro si falta alguna columna que el transformador necesita."""
    faltantes = [c for c in columnas if c not in X.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas en el DataFrame: {faltantes}")


class FeaturesDeFecha(BaseEstimator, TransformerMixin):
    """
    Crea variables a partir de la fecha de la carrera.

    - antiguedad_en_dias: días entre la carrera y una fecha de referencia FIJA.
      Se usa una fecha fija (y no datetime.now()) para que el resultado sea
      reproducible: si no, el mismo dataset daría valores distintos cada día.
    - mes_carrera: mes del año (1 a 12). Sirve como señal de la parte de la
      temporada en la que se corrió.
    """

    def __init__(self, columna_fecha="date", fecha_referencia="2026-10-10"):
        self.columna_fecha = columna_fecha
        self.fecha_referencia = fecha_referencia

    def fit(self, X, y=None):
        # No aprende nada de los datos: la fecha de referencia es un parámetro.
        _verificar_columnas(X, [self.columna_fecha])
        return self

    def transform(self, X):
        _verificar_columnas(X, [self.columna_fecha])
        X_copy = X.copy()  # para no modificar el DataFrame original
        fechas = pd.to_datetime(X_copy[self.columna_fecha])
        referencia = pd.to_datetime(self.fecha_referencia)
        X_copy["antiguedad_en_dias"] = (referencia - fechas).dt.days
        X_copy["mes_carrera"] = fechas.dt.month
        return X_copy


class FeaturesDeCarrera(BaseEstimator, TransformerMixin):
    """
    Crea variables que describen cómo le fue al piloto en la carrera.

    - posiciones_ganadas: largada menos llegada. Positivo = ganó lugares.
      Queda en NaN si no clasificó (no tiene posición de llegada), porque
      decir que "ganó 0 lugares" sería un dato engañoso.
    - termino_en_puntos: 1 si llegó entre los 10 primeros, 0 si no.
    - no_clasifico: 1 si no tiene posición final (abandono o descalificación).
    """

    def __init__(self, columna_largada="grid", columna_llegada="position"):
        self.columna_largada = columna_largada
        self.columna_llegada = columna_llegada

    def fit(self, X, y=None):
        # Reglas fijas, no hay nada que aprender.
        _verificar_columnas(X, [self.columna_largada, self.columna_llegada])
        return self

    def transform(self, X):
        _verificar_columnas(X, [self.columna_largada, self.columna_llegada])
        X_copy = X.copy()
        # to_numeric + float: así los nulos (NaN o None) quedan todos como NaN
        # y las comparaciones no fallan con el tipo Int64 de pandas.
        largada = pd.to_numeric(X_copy[self.columna_largada], errors="coerce").astype(float)
        llegada = pd.to_numeric(X_copy[self.columna_llegada], errors="coerce").astype(float)

        X_copy["posiciones_ganadas"] = largada - llegada
        X_copy["termino_en_puntos"] = (llegada <= 10).astype(int)  # NaN <= 10 da False
        X_copy["no_clasifico"] = llegada.isna().astype(int)
        return X_copy


class PromedioPorCategoria(BaseEstimator, TransformerMixin):
    """
    Agrega el promedio de una columna numérica por categoría
    (por defecto, puntos promedio por escudería).

    Este es el transformador que APRENDE en fit:
    - fit: calcula el promedio por categoría usando solo los datos que recibe
      (train) y lo guarda en promedios_. También guarda la media global.
    - transform: busca el promedio de cada fila en ese diccionario. Si aparece
      una categoría que no estaba en train, usa la media global de train.

    Importante: usar fit_transform solo sobre train y transform sobre test.
    Si se hiciera fit sobre test, el transformador vería datos que no debería.
    """

    def __init__(self, columna_categoria="constructor", columna_valor="points",
                 nombre_salida="promedio_puntos_escuderia"):
        self.columna_categoria = columna_categoria
        self.columna_valor = columna_valor
        self.nombre_salida = nombre_salida

    def fit(self, X, y=None):
        _verificar_columnas(X, [self.columna_categoria, self.columna_valor])
        categoria = X[self.columna_categoria].astype(object)
        valor = pd.to_numeric(X[self.columna_valor], errors="coerce")

        # Los atributos que se aprenden terminan en "_" (convención de sklearn)
        self.promedios_ = valor.groupby(categoria).mean().to_dict()
        self.media_global_ = float(valor.mean())
        return self

    def transform(self, X):
        check_is_fitted(self, ["promedios_", "media_global_"])
        _verificar_columnas(X, [self.columna_categoria])
        X_copy = X.copy()
        categoria = X_copy[self.columna_categoria].astype(object)
        # Se pasa a object antes del map porque si la columna es "category"
        # no se puede rellenar con un valor que no sea una de sus categorías.
        X_copy[self.nombre_salida] = (
            categoria.map(self.promedios_).astype(float).fillna(self.media_global_)
        )
        return X_copy
