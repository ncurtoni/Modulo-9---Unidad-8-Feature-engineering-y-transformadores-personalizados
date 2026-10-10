# Feature engineering con transformadores personalizados - Colapinto

Ejercicio de la Unidad 8 de Data Science II (Coderhouse, comisión 90545). Partiendo del dataset de resultados de carrera de Franco Colapinto (el mismo del pipeline ETL anterior, extraído de la API Jolpica-F1), se crean variables nuevas con transformadores propios compatibles con Scikit-Learn.

## Estructura

```
custom_transformers.py        # las 3 clases (BaseEstimator + TransformerMixin)
main.py                       # flujo completo: carga, diagnóstico, split, fit/transform, chequeos
test_custom_transformers.py   # tests con pytest
requirements.txt
.gitignore
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py
pytest
```

En Windows con VS Code, si `pip` o `python` no se reconocen, usar `py -m pip install -r requirements.txt` y `py main.py`.

`main.py` busca los datos en este orden:

1. Un archivo `dataset.csv` en la misma carpeta (por ejemplo, el `data/raw/dataset.csv` que genera el ETL).
2. La API de Jolpica-F1, igual que en el ETL.
3. Si no hay ni archivo ni conexión, genera un dataset simulado con las mismas columnas, solo para que el script corra en cualquier máquina. Esos valores no son resultados reales y el script avisa cuando los usa.

## Qué problema resuelve

El modelo no entiende por sí solo qué significa una fecha ni la diferencia entre largar 18 y llegar 12. Y si el cálculo de las variables se repite a mano en cada notebook, es fácil que train y test se procesen distinto. Los transformadores encapsulan cada regla en una clase reutilizable, y la que necesita aprender algo de los datos lo hace solo en `fit`.

## Variables nuevas y por qué aportan

| Variable | Transformador | Qué aporta al modelo |
|---|---|---|
| `antiguedad_en_dias` | `FeaturesDeFecha` | Días desde la carrera hasta una fecha de referencia fija. Permite que el modelo dé más peso a las carreras recientes si el rendimiento del piloto cambió con el tiempo. |
| `mes_carrera` | `FeaturesDeFecha` | Mes del año. Resume en qué parte de la temporada se corrió (circuitos, condiciones y estado del auto cambian a lo largo del año). |
| `posiciones_ganadas` | `FeaturesDeCarrera` | Largada menos llegada. Mide el rendimiento en carrera descontando el efecto de la clasificación. Queda en NaN si no clasificó, para no inventar un 0. |
| `termino_en_puntos` | `FeaturesDeCarrera` | 1 si llegó entre los 10 primeros. Es un indicador simple y robusto del resultado. |
| `no_clasifico` | `FeaturesDeCarrera` | 1 si no tiene posición final. Conserva el hecho de que el dato faltaba, en lugar de esconderlo (la ausencia también informa). |
| `promedio_puntos_escuderia` | `PromedioPorCategoria` | Puntos promedio de la escudería. Resume en un número qué tan competitivo era el auto, sin tener que usar el nombre como categoría. |

## Data leakage

Solo `PromedioPorCategoria` aprende parámetros (el promedio por escudería y la media global). Por eso:

- `main.py` divide en train y test **antes** de ajustar nada.
- Train usa `fit_transform` y test usa únicamente `transform`.
- Si en test aparece una escudería que no estaba en train, se le asigna la media global **de train**.
- El script comprueba con `assert` que los promedios aprendidos salen solo de train, y los tests verifican que `transform` sobre otro test no cambia lo aprendido.

`FeaturesDeFecha` y `FeaturesDeCarrera` aplican reglas fijas, así que su `fit` no aprende nada. La fecha de referencia de `antiguedad_en_dias` es un parámetro fijo (no `datetime.now()`) para que el resultado sea reproducible.

Los tres transformadores usan `X.copy()` dentro de `transform`, así que el DataFrame original no se modifica.

## Nulos

Antes de crear variables, `main.py` imprime los nulos por columna con `df.isna().sum()`. `isna()` detecta tanto `NaN` (el nulo float de numpy, que es el que queda en las columnas numéricas) como `None` (el nulo de Python, que puede quedar en columnas de texto). En este dataset el único nulo esperado está en `position`: no es un error de carga, es que el piloto no clasificó, por eso no se rellena y se marca con `no_clasifico`.

## Pipeline

Al final de `main.py` se arma un `Pipeline` con los tres transformadores y se comprueba que da exactamente el mismo resultado que aplicarlos uno por uno.

## Salida de ejemplo (datos simulados)

```
Train: 32 filas | Test: 9 filas

Promedios aprendidos SOLO con train: {'Alpine F1 Team': 0.24, 'Williams': 2.14}
Chequeo de leakage OK: los parámetros salen solo de train y los DataFrames originales no cambiaron.
El Pipeline da exactamente el mismo resultado que aplicar los pasos a mano.
```
