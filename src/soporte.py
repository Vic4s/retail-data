import pandas as pd
import numpy as np
from IPython.display import display
from pathlib import Path
import unicodedata


    
def cargar_df_list_desde_directorio_csv(dir_path, patron="*.csv", imprimir_info=True, devolver_detalle=False):
    """
    Busca CSVs en un directorio, intenta leer cada archivo con cargar_todos_csv
    y devuelve una lista de DataFrames.

    Args:
        dir_path (str | Path): Directorio donde buscar CSVs.
        patron (str): Patrón de búsqueda (por defecto "*.csv").
        imprimir_info (bool): Si True, imprime resumen de archivos encontrados.
        devolver_detalle (bool): Si True, devuelve también rutas, encodings y separadores.

    Returns:
        list[pd.DataFrame] | tuple: 
            - Si devolver_detalle=False: lista de DataFrames
            - Si devolver_detalle=True: (df_list, paths, encodings, separadores)
    """
    base = Path(dir_path)
    csv_paths = list(base.glob(patron))

    if imprimir_info:
        print(f"[INFO] Encontrados {len(csv_paths)} CSV:")
        for p in csv_paths:
            print("   -", p.name)

    if not csv_paths:
        raise FileNotFoundError(
            f"No se encontraron CSV en {base.resolve()} con patrón '{patron}'. ¿Ruta correcta?"
        )

    df_list = []
    paths = []
    encs = []
    seps = []

    for path in csv_paths:
        df, enc, sep = cargar_todos_csv(path)
        df_list.append(df)
        paths.append(path)
        encs.append(enc)
        seps.append(sep)

    if devolver_detalle:
        return df_list, paths, encs, seps

    return df_list



def cargar_todos_csv(path):
    """
    Intenta leer un CSV probando varios separadores y encodings.
    Devuelve (df, encoding, sep). Lanza ValueError si no puede leer.
    """

    archivo = []

    for enc in ("utf-8-sig", "utf-8", "latin-1", "utf-16"):

        for sep in (",", ";", "|", "\t"):

            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                
                if df.shape[1] > 1: # al menos 2 columnas
                    print(f"[OK] {path.name}: encoding={enc}, sep='{sep}', shape={df.shape}")
                    return df, enc, sep 
                else:
                    archivo.append((enc, sep, "solo 1 columna"))
            except Exception as e:
                archivo.append((enc, sep, str(e)))

    # En caso de error, mostrar intentos
    print(f"[ERROR] No pude leer {path.name}. Intentos:")

    for enc, sep, err in archivo[:10]:  # mostrar algunos intentos
        print(f"   enc={enc}, sep='{sep}' -> {err}")

    raise ValueError(f"No se pudo leer {path}")




def normalizar_nombres_columnas(name: str) -> str:
    """
    Normaliza el nombre de una columna.
    """
    name = name.strip()
    name = name.lower()
    name = name.replace(" ", "_")
    name = name.replace("%", "pct")
    name = name.replace("-", "_")
    return name


def normalizar_columnas_df_list(df_list, mostrar_cambios=True):
    """
    Normaliza los nombres de columnas de cada DataFrame en df_list usando normalizar_nombres_columnas().

    Parámetros:
    - df_list: lista de pandas.DataFrame
    - mostrar_cambios: si True, imprime las columnas antes/después y los cambios detectados

    Retorna:
    - lista de DataFrames con columnas normalizadas
    """
    df_list_norm = []  # lista de dataframes normalizados

    for i, df in enumerate(df_list):
        old_cols = list(df.columns)
        d2 = df.rename(columns=lambda c: normalizar_nombres_columnas(str(c)))
        new_cols = list(d2.columns)

        if mostrar_cambios and old_cols != new_cols:
            print(f"[INFO] Normalizadas columnas en DF {i}:")
            cambios = list(zip(old_cols, new_cols))
            for old, new in cambios:
                if old != new:
                    print(f"   -   '{old}' -> '{new}'")

        df_list_norm.append(d2)

    return df_list_norm



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



def nulos_df (df, ordenar_desc=False):
    """ 
    Devuelve un Dataframe con las columnas y el porcentaje de nulos de cada columna.
    """

    df_nulos = pd.DataFrame({
        'Nulos': df.isnull().sum(),
        '% Nulos': round(((df.isnull().sum() / len(df) ) *100),2)
    })

    if ordenar_desc:
        df_nulos = df_nulos.sort_values('Nulos', ascending=False)

    return df_nulos



def resumen_df(df):
    """
    Devuelve un DataFrame con un resumen completo de cada columna del DataFrame dado.
    
    Muestra para cada columna:
    - Tipo de dato
    - Total de valores
    - Valores únicos
    - Porcentaje de nulos
    - Cantidad de valores duplicados
    
    Args:
        df (pd.DataFrame): DataFrame de entrada
        
    Returns:
        pd.DataFrame: DataFrame con el resumen de cada columna
    """
    return pd.DataFrame({
        'Columna': df.columns,
        'Tipo': [str(df[col].dtype) for col in df.columns],
        'Total_Valores': [len(df[col]) for col in df.columns],
        'Valores_Unicos': [df[col].nunique() for col in df.columns],
        '%_Nulos': [round((df[col].isnull().sum() / len(df)) * 100, 2) 
                    for col in df.columns],
        'Duplicados': [df[col].duplicated().sum() for col in df.columns]
    })



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

    

def limpiar_columna_texto(df, nombre_col, col_no_cambiar=None) :
    """
    Limpia y normaliza in-place una columna de texto en un DataFrame,
    y devuelve cuántos valores han sido modificados.

    Transformaciones aplicadas:
      1. Convertir a string.
      2. Eliminar espacios al principio y al final.
      3. Colapsar múltiples espacios internos en uno solo.
      4. Eliminar puntos “.”.
      5. Normalizar caracteres Unicode (acentos, eñes…) a ASCII.
      6. Convertir a mayúsculas.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna a limpiar.
    nombre_col : str
        Nombre de la columna de texto a estandarizar.
    col_no_cambiar : list
        Lista de columnas que no se deben limpiar.

    Retorna
    -------
    int
        Número de celdas cuyo valor ha cambiado tras la limpieza.
    """

    if nombre_col in col_no_cambiar:

        # Guardamos copia para comparar después
        valor_original = df[nombre_col].astype(str).copy()

        valor_modificado = df[nombre_col].astype(str)
        # 1) strip
        valor_modificado = valor_modificado.str.strip()
        # 2) colapsar espacios
        valor_modificado = valor_modificado.str.replace(r'\s+', ' ', regex=True)
        # 3) quitar puntos
        servalor_modificadoie = valor_modificado.str.replace('.', '', regex=False)
        # 4) normalizar Unicode → ASCII
        valor_modificado = valor_modificado.map(lambda s: unicodedata.normalize('NFKD', s))
        valor_modificado = valor_modificado.str.encode('ascii', errors='ignore').str.decode('utf-8')
        # 5) Title
        valor_modificado = valor_modificado.str.title()
        # 6) Denominaciones sociales en mayúsculas
        valor_modificado = valor_modificado.str.capitalize()

        # Sobrescribimos en el DataFrame
        df[nombre_col] = valor_modificado

    # Contamos cuántos han cambiado
    # (se compara string a string)
    diffs = (valor_original != valor_modificado).sum()
    print(f"{diffs} valores actualizados en la columna '{nombre_col}'")


    

  