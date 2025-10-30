import pandas as pd
import numpy as np
from IPython.display import display


def resumen_valores_columna(df, columna):
    """
    Devuelve un resumen con los valores únicos y el porcentaje de cada valor (incluyendo nulos) de una columna.

    Parámetros:
    - df: DataFrame de entrada
    - columna: nombre de la columna a analizar (str)

    Retorna:
    - Un DataFrame con columnas: 'Cantidad' y '%'
    """
    total = df[columna].shape[0]

    cont = df[columna].value_counts(dropna=False).reset_index()
    cont.columns = [columna, 'Cantidad']
    cont['%'] = round((cont['Cantidad'] / total) * 100, 2)

    return cont


def mostrar_todos_valores_columna(df, columna, max_filas=None):
    """
    Muestra por pantalla el resumen de valores únicos de una columna
    usando una función resumen, mostrando todas (o n) filas en pantalla.

    Parámetros:
    - df: DataFrame de entrada.
    - columna: str, nombre de la columna a resumir.
    - funcion_resumen: función que recibe (df, columna) y devuelve un DataFrame resumen.
    - max_filas: int o None, máximo de filas a mostrar (None para mostrar todas).

    Ejemplo de uso:
        mostrar_resumen_columna(df, 'Cliente')
        mostrar_resumen_columna(df_2, "Cliente", max_filas=50)
    """
    if max_filas is None:
        with pd.option_context('display.max_rows', None):
            display(resumen_valores_columna(df, columna))
    else:
        with pd.option_context('display.max_rows', max_filas):
            display(resumen_valores_columna(df, columna))



def nulos_df (df):

    df_nulos = pd.DataFrame({
        'Nulos': df.isnull().sum(),
        '% Nulos': round(((df.isnull().sum() / len(df) ) *100),2)
    })

    return df_nulos





def convertir_columnas_datetime(df, columnas, formato="%d/%m/%Y"):
    """
    Convierte columnas a tipo datetime con control de errores.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        columnas (list): Lista de nombres de columnas a convertir.
        formato (str): Formato esperado. Por defecto: "%d/%m/%Y".

    Returns:
        pd.DataFrame: DataFrame con columnas convertidas a datetime.
    """
    for col in columnas:
        try:
            df[col] = pd.to_datetime(df[col], format=formato, errors='coerce')

            errores = df[col].isna().sum()
            convertidos = df[col].notna().sum()

            total = len(df[col])
            porcentaje_convertidos = round((convertidos / total) * 100, 2)
            print(f"En '{col}' hay {convertidos} valores convertidos ({porcentaje_convertidos}%).")

            porcentaje_errores = round((errores / total) * 100, 2)
            print(f"En '{col}' hay {errores} valores no convertidos ({porcentaje_errores}%).")

        except Exception as e:
            print(f"Error al convertir la columna '{col}': {e}")

            



def convertir_columna_bool(df, columna):
    """
    Reemplaza en df[columna] los valores 'si'/'sí' por True y 'no' por False,
    y convierte toda la columna al tipo booleano.

    Args:
      df      : DataFrame sobre el que trabajamos (modifica in-place).
      columna : nombre de la columna a convertir.

    Returns:
      None (modifica df directamente).
    """

    # Comprobar que la columna existe
    if columna not in df.columns:
        print(f"La columna '{columna}' no existe en el DataFrame.")
        return

    # Normalizar a minúsculas y quitar espacios al inicio y fin
    df[columna] = df[columna].astype(str).str.strip().str.lower()

    # Mapear los valores a booleanos
    df[columna] = df[columna].map({
        'si': True,
        'sí': True,
        'no': False
    })

    # Convertir el tipo de la columna a bool
    df[columna] = df[columna].astype('boolean') # usamos .astype('boolean') en vez de .astype(bool) porque .astype('boolean') permite mantener los nulos (con <NA>) sin convertilos a False automáticamente  

    print(f"Columna '{columna}' convertida a booleano.")

    

 