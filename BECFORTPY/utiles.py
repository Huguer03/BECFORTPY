import numpy as np 

def redondear_vectorizado(df_col_valor, df_col_error):
    posicion = -np.floor(np.log10(np.abs(df_col_error))).astype(int)
    v_redondo = [round(v, p) for v, p in zip(df_col_valor, posicion)]
    e_redondo = [round(e, p) for e, p in zip(df_col_error, posicion)]
    return v_redondo, e_redondo

def redondear_escalares(valor, error):
    if error == 0:
        return valor, 0
    posicion = -int(np.floor(np.log10(abs(error))))
    return round(valor, posicion), round(error, posicion)
