import numpy as np
from numpy import array, linspace, diag, sqrt, clip, newaxis, exp,  tanh
from numpy.linalg import lstsq
from pandas import DataFrame, Series, to_numeric, read_csv
from scipy.optimize import curve_fit
from pathlib import Path
from matplotlib.pyplot import errorbar, gca, figure, title, xlabel, ylabel, legend, grid, show

EXTENSION = ".CSV"

QUADRATIC_MODEL = "quadratic"
TANH_MODEL = "tanh"

# Voltaje de offset para corregir posibles desviaciones en la medición
V_OFFSET = 0.0
# Ventana de voltaje para filtrar datos (None para no filtrar)
V_VENTANA = None
# Límite inferior para filtrar datos (None para no filtrar)
LI_CORRIENTE = None
# Límite superior para filtrar datos (None para no filtrar)
LS_CORRIENTE = None
# Límite inferior para filtrar datos de V_GS (None para no filtrar)
LI_V_GS = -0.55
LS_V_GS = -0.149          # Límite superior para filtrar datos de V_G

V_DD_LAB = 15.0
RESISTENCIA = 221.0  # Resistencia de shunt R1
RES_CARGA = 1000.0  # Resistencia de carga para cálculo de ganancia

RUTA_TRANSISTORES = Path("./transistores")
RUTA_RESULTADOS = Path("./resultados")

CIRCUITOS_PRUEBA = [
    # 1. FIXED BIAS (3 valores de R_D)
    {'nombre': 'Fixed Bias 1', 'tipo': 'fixed', 'V_GG': 1.5, 'R_D': 1000},
    {'nombre': 'Fixed Bias 2', 'tipo': 'fixed', 'V_GG': 1.5, 'R_D': 2200},
    {'nombre': 'Fixed Bias 3', 'tipo': 'fixed', 'V_GG': 1.5, 'R_D': 3300},

    # 2. SELF BIAS (3 pares de R_D y R_S)
    {'nombre': 'Self Bias 1', 'tipo': 'self', 'R_D': 2200, 'R_S': 470},
    {'nombre': 'Self Bias 2', 'tipo': 'self', 'R_D': 3300, 'R_S': 1000},
    {'nombre': 'Self Bias 3', 'tipo': 'self', 'R_D': 4700, 'R_S': 2200},

    # 3. VOLTAGE DIVIDER (3 conjuntos ajustados a máx 500k)
    {'nombre': 'Divider Bias 1', 'tipo': 'divider',
     'R_D': 2200, 'R_S': 1000, 'R_1': 220e3, 'R_2': 22e3},
    {'nombre': 'Divider Bias 2', 'tipo': 'divider',
     'R_D': 3300, 'R_S': 2200, 'R_1': 100e3, 'R_2': 10e3},
    {'nombre': 'Divider Bias 3', 'tipo': 'divider',
     'R_D': 4700, 'R_S': 3300, 'R_1': 330e3, 'R_2': 33e3},
]


def filtrar_extension(directorio, extension=EXTENSION):
    try:
        ruta = Path(directorio)
        patron = f"*{extension}"
        archivos = [archivo.name.removesuffix(
            EXTENSION) for archivo in ruta.glob(patron) if archivo.is_file()]
        return archivos
    except FileNotFoundError:
        print(
            f"Error: El directorio '{directorio}' no existe o la ruta es incorrecta.")
        exit(1)
    except Exception as e:
        print(f"Error filtrar_extension: {e}")
        exit(1)


def leer(directorio, skiprows=0, header=None):
    try:
        return read_csv(directorio, skiprows=skiprows, header=header)
    except FileNotFoundError:
        print(f"Error: El archivo '{directorio}' no fue encontrado.")
        return DataFrame()
    except Exception as e:
        print(f"Ocurrio un error al leer el archivo CSV: {e}")
        return DataFrame()


def limpiar(directorio, col_t_1=0, col_y_1=1, col_y_2=2):
    try:
        df = leer(directorio)
        columnas_a_limpiar = [col_t_1, col_y_1, col_y_2]
        for col in columnas_a_limpiar:
            df[col] = to_numeric(df[col], errors="coerce")
        df_limpio = df.dropna(subset=columnas_a_limpiar).copy()
        if col_t_1 in df_limpio.columns:
            df_limpio[col_t_1] = (df_limpio[col_t_1] -
                                  df_limpio[col_t_1].iloc[0])
        return df_limpio
    except FileNotFoundError:
        print(f"El archivo '{directorio}' no fue encontrado.")
        exit(1)
    except Exception as ex:
        print(f"Error limpiar: {ex}")
        exit(1)


def obtener_datos(nombre_archivo, v_offset, v_ventana=None, col_t=0, col_va=1, col_vb=2, extension=EXTENSION):
    df: DataFrame = limpiar(
        f"{nombre_archivo}{extension}", col_t_1=col_t, col_y_1=col_va, col_y_2=col_vb)
    li = None
    ls = None
    if v_ventana is not None:
        ls = df.iloc[:, col_vb].max()
        li = ls - v_ventana
    return df.iloc[:, col_t], df.iloc[:, col_va], df.iloc[:, col_vb] + v_offset, li, ls


def filtrar_datos(x, y, li=None, ls=None):
    # Handle both numpy arrays and pandas Series
    if isinstance(x, Series):
        mascara = Series([True] * len(x), index=y.index)
    else:
        # For numpy arrays, create a boolean mask
        mascara = array([True] * len(x), dtype=bool)

    if li is not None:
        mascara &= (x >= li)
    if ls is not None:
        mascara &= (x <= ls)
    return x[mascara], y[mascara]


def graficar_columnas(x, y, tit, nom_x, nom_y, label=None, markersize=0.1, fmt="-", alpha=1.0, yerr=None, names=None):
    figure(figsize=(10, 5))
    if not isinstance(label, list):
        label = [label] * len(x)
    if not isinstance(markersize, list):
        markersize = [markersize] * len(x)
    if not isinstance(fmt, list):
        fmt = [fmt] * len(x)
    if not isinstance(alpha, list):
        alpha = [alpha] * len(x)
    if not isinstance(yerr, list):
        yerr = [yerr] * len(x)
    if not isinstance(names, list):
        names = [names] * len(x)

    for x_i, y_i, label_i, markersize_i, fmt_i, alpha_i, yerr_i, names_i in zip(x, y, label, markersize, fmt, alpha, yerr, names):
        x_i, y_i = array(x_i), array(y_i)
        errorbar(x=x_i, y=y_i, label=label_i, markersize=markersize_i, fmt=fmt_i,
                 ecolor="black", elinewidth=0.5, alpha=alpha_i, yerr=yerr_i, capsize=3)
        ax = gca()
        if names_i is not None:
            for x_j, y_j, names_j in zip(x_i, y_i, names_i):
                ax.annotate(names_j, xy=(x_j, y_j), xytext=(
                    0, 2), textcoords="offset points", fontsize=8, ha="center", va="bottom")
    title(tit)
    xlabel(nom_x)
    ylabel(nom_y)
    legend()
    grid()
    show()


def graficar_voltajes(t, v_a, v_b, nombre, v_gs):
    graficar_columnas(
        [t, t],
        [v_a, v_b],
        f"Voltajes Va y Vb {nombre} a V_GS={v_gs}V",
        "Tiempo (s)",
        "Amplitud (V)",
        ["Va", "Vb"])


def graficar_corriente(v_b, i, v_ds, i_d, i_d_std, nombre, v_gs):
    graficar_columnas(
        [v_b, v_ds],
        [i, i_d],
        f"Corriente vs Voltaje {nombre} a V_GS={v_gs}V",
        "Voltaje (V)",
        "Corriente (A)",
        ["I_D", "I_D_promedio"],
        [1, 10],
        ["o", "-"],
        [0.2, 1.0],
        [None, i_d_std])


def graficar_regresion(x, y, ec_fit, nombre):
    graficar_columnas(
        x=x, y=y,
        tit=f"Curva de Transferencia (I_D vs V_GS) - {nombre}",
        nom_x="Voltaje V_GS (V)", nom_y="Corriente de Saturación I_D (A)",
        label=["Datos extraídos (Estables)",
               f"Ajuste no emergencial: {ec_fit[0]}"],
        markersize=[4, 0.1, 0.1], fmt=["o", "--", "-"]
    )


def graficar_familia_de_curvas(x, y, nombre):
    graficar_columnas(
        x=x, y=y,
        tit=f"Familia de Curvas Características (I_D vs V_DS) - {nombre}",
        nom_x="Voltaje Drain-Source V_DS (V)", nom_y="Corriente I_D (A)",
        label=None,
        markersize=0.1,
        fmt="-o"
    )


def valores_representativos(x, y, num_valores=100, min_puntos=20):
    x_i = x.min()
    x_s = x.max()
    dx = (x_s - x_i) / num_valores
    x_umbral, y_promedio, y_std = [], [], []
    for j in range(num_valores):
        bin_start = x_i + j * dx
        bin_end = x_i + (j + 1) * dx
        y_cercano = y[(x >= bin_start) & (x < bin_end)]
        if len(y_cercano) >= min_puntos:
            x_umbral.append((bin_start + bin_end) / 2)
            y_promedio.append(y_cercano.mean())
            std_val = y_cercano.std()
            y_std.append(std_val if std_val > 0 else 1e-6)
    return array(x_umbral), array(y_promedio), array(y_std)


def regresion_no_emergencial(x, y, a, b, c, y_std=None, resolucion_regresion=1000, model=None, limites=None):
    def quadratic(x, a, b, c):
        return a * (1 - x / b)**2 + c

    def tanh_model(x, a, b, c):
        return a * tanh(b * x) * (1 + c * x)
    if model == QUADRATIC_MODEL:
        modelo = quadratic
    elif model == TANH_MODEL:
        modelo = tanh_model

    x_fit = linspace(x.min(), x.max(), resolucion_regresion)
    kwargs = {
        'p0': [a, b, c],
        'sigma': y_std,
        'absolute_sigma': True,
        'method': 'lm' if limites is None else 'trf',
        'maxfev': 10000
    }
    if limites is not None:
        kwargs['bounds'] = limites

    (a_, b_, c_), pcov = curve_fit(modelo, x, y, **kwargs)
    return ([x_fit, modelo(x_fit, a_, b_, c_)], (a_, b_, c_), sqrt(diag(pcov)))


def caracterizar_transistor(transistor_path, res=RESISTENCIA, v_offset=0.0, v_ventana=None, li=None, ls=None, zona_saturacion=0.2):
    nombre_transistor = transistor_path.name
    nombres = filtrar_extension(transistor_path)
    transistor_dict = {}

    v_gs_list, i_dss_list, label_list = [], [], []
    v_ds_curves, i_d_curves = [], []

    for nombre in nombres:
        v_gs = float(nombre.upper().replace("V", ""))
        ruta_completa = str(transistor_path / nombre)

        t, v_a, v_b, _, _ = obtener_datos(ruta_completa, v_offset, v_ventana)
        # graficar_voltajes(t, v_a, v_b, nombre_transistor, v_gs)
        i = (v_a - v_b) / res
        filt_i, filt_v_b = filtrar_datos(i, v_b, li=li, ls=ls)
        v_ds, i_d, i_d_std = valores_representativos(filt_v_b, filt_i)
        # graficar_corriente(filt_v_b, filt_i, v_ds, i_d, i_d_std, nombre_transistor, v_gs)

        i_dss = filt_i[filt_v_b >=
                       filt_v_b.max() * (1 - zona_saturacion)].mean()

        a_ini = max(i_d)
        b_ini = 2.0
        c_ini = 0.01

        (v_ds_fit_ne, i_d_fit_ne), _, _ = regresion_no_emergencial(
            v_ds,
            i_d,
            a_ini, b_ini, c_ini,
            y_std=i_d_std,
            model=TANH_MODEL
        )
        # graficar_regresion([v_ds, v_ds_fit_ne], [i_d, i_d_fit_ne], nombre_transistor, v_gs)

        v_gs_list.append(v_gs)
        i_dss_list.append(i_dss)
        label_list.append(f"V_GS={v_gs:.6f}V")
        v_ds_curves.append(v_ds_fit_ne)
        i_d_curves.append(i_d_fit_ne)

    transistor_dict.update({nombre_transistor: {
        "V_GS": v_gs_list,
        "I_DSS": i_dss_list,
        "Labels": label_list,
        "V_DS_CURVES": v_ds_curves,
        "I_D_CURVES": i_d_curves
    }})
    return transistor_dict


def resolver_polarizacion_teorica(i_dss, v_p, v_dd, config):
    tipo = config['tipo']
    r_d = config.get('R_D', 0)
    r_s = config.get('R_S', 0)

    v_g = 0.0
    v_gs = 0.0
    i_d = 0.0

    if tipo == 'fixed':
        # Polarización Fija
        v_gg = config.get('V_GG', 0)
        v_g = -v_gg  # Asumiendo que V_GG se conecta con el negativo al Gate
        v_gs = v_g
        # Si V_GS es más negativo que V_P, el transistor está en corte
        if v_gs < v_p:
            i_d = 0
        else:
            i_d = i_dss * (1 - v_gs / v_p)**2

    elif tipo in ['self', 'divider']:
        # Autopolarización o Divisor de Voltaje
        if tipo == 'divider':
            r_1 = config.get('R_1', 1)
            r_2 = config.get('R_2', 1)
            v_g = v_dd * (r_2 / (r_1 + r_2))
        else:  # self
            v_g = 0.0

        # Reordenando malla y Shockley queda una ecuación cuadrática a*V_GS^2 + b*V_GS + c = 0
        a = i_dss / (v_p**2)
        b = (1.0 / r_s) - (2 * i_dss / v_p)
        c = i_dss - (v_g / r_s)

        discriminante = b**2 - 4*a*c
        if discriminante < 0:
            return None  # No hay solución real

        # Raíces de la ecuación
        v_gs_1 = (-b + sqrt(discriminante)) / (2*a)
        v_gs_2 = (-b - sqrt(discriminante)) / (2*a)

        # Físicamente V_GS debe estar entre V_P (corte) y el voltaje máximo de compuerta
        validos = [v for v in (v_gs_1, v_gs_2) if v_p <= v <= v_g + 0.5]

        if not validos:
            return None

        v_gs = max(validos)  # Tomar la raíz con sentido físico
        i_d = i_dss * (1 - v_gs / v_p)**2

    v_s = i_d * r_s
    v_d = v_dd - (i_d * r_d)
    v_ds = v_d - v_s

    return {
        "V_GS": v_gs, "I_D": i_d, "V_DS": v_ds,
        "V_D": v_d, "V_G": v_g, "V_S": v_s,
        "I_S": i_d, "I_G": 0.0
    }


if __name__ == "__main__":

    if not RUTA_TRANSISTORES.exists():
        print("La carpeta './transistores' no existe.")
        exit()

    for transistor_path in RUTA_TRANSISTORES.iterdir():
        nombre_transistor = transistor_path.name
        if not transistor_path.is_dir():
            continue

        transistor_dict = caracterizar_transistor(
            transistor_path, v_offset=V_OFFSET, v_ventana=V_VENTANA, li=LI_CORRIENTE, ls=LS_CORRIENTE, zona_saturacion=0.2)
        v_gs_list = transistor_dict[nombre_transistor]["V_GS"]
        i_dss_list = transistor_dict[nombre_transistor]["I_DSS"]
        label_list = transistor_dict[nombre_transistor]["Labels"]
        v_ds_curves = transistor_dict[nombre_transistor]["V_DS_CURVES"]
        i_d_curves = transistor_dict[nombre_transistor]["I_D_CURVES"]

        graficar_familia_de_curvas(
            v_ds_curves,
            i_d_curves,
            nombre_transistor
        )

        v_gs_arr = array(v_gs_list)
        i_dss_arr = array(i_dss_list)
        idx_sort = v_gs_arr.argsort()
        v_gs_arr = v_gs_arr[idx_sort]
        i_dss_arr = i_dss_arr[idx_sort]

        v_gs_filt, i_dss_filt = filtrar_datos(
            v_gs_arr, i_dss_arr, li=LI_V_GS, ls=LS_V_GS)

        a_ini = max(i_dss_filt)
        b_ini = min(v_gs_filt)
        c_ini = 0.0

        (v_gs_fit, i_dss_fit), (a_ne, b_ne, c_ne), (a_std, b_std, c_std) = regresion_no_emergencial(
            v_gs_filt,
            i_dss_filt,
            a_ini, b_ini, c_ini,
            model=QUADRATIC_MODEL
        )
        graficar_regresion(
            [v_gs_filt, v_gs_fit],
            [i_dss_filt, i_dss_fit],
            [f"I_D = {a_ne:.2e} * (1 - V_GS / {b_ne:.6f})^2 + {c_ne:.2e}"],
            nombre_transistor
        )

        i_d_q = a_ne / 4.0
        v_gs_q = b_ne / 2.0

        std_i_d_q = a_std / 4.0
        std_v_gs_q = b_std / 2.0

        g_m = (-2 * a_ne / b_ne) * (1 - v_gs_q / b_ne)
        A_v = -g_m * RES_CARGA

        idx_vgs_max = v_gs_arr.argmax()
        v_ds_ref = v_ds_curves[idx_vgs_max]
        i_d_ref = i_d_curves[idx_vgs_max]

        limite_idx = int(len(v_ds_ref) * 0.8)
        if len(v_ds_ref) > limite_idx and (i_d_ref[-1] - i_d_ref[limite_idx]) > 1e-6:
            delta_v = v_ds_ref[-1] - v_ds_ref[limite_idx]
            delta_i = i_d_ref[-1] - i_d_ref[limite_idx]
            r_d_calculado = delta_v / delta_i
            lambda_val = 1 / \
                (r_d_calculado * i_dss_arr[idx_vgs_max]
                 ) if i_dss_arr[idx_vgs_max] > 0 else 0.0
        else:
            r_d_calculado = float('inf')
            lambda_val = 0.0

        # === GUARDADO DEL REPORTE ===
        ruta_archivo_salida = RUTA_RESULTADOS / \
            f"resultados_{nombre_transistor}.txt"
        with open(ruta_archivo_salida, "w", encoding="utf-8") as f:
            f.write(f"=== REPORTE DEL TRANSISTOR: {nombre_transistor} ===\n\n")
            f.write("--- PUNTO DE TRABAJO ÓPTIMO (PEQUEÑA SEÑAL) ---\n")
            f.write(f"V_GSQ = {v_gs_q:.6f} V  ± {std_v_gs_q:.6f} V\n")
            f.write(f"I_DQ  = {i_d_q:.6f} A  ± {std_i_d_q:.6f} A\n\n")

            f.write("--- AMPLIFICACIÓN TEÓRICA ---\n")
            f.write(f"Transconductancia (g_m): {g_m:.6f} S\n")
            f.write(f"Ganancia de Voltaje (A_v): {A_v:.6f} V/V\n\n")

            # === NUEVO: ESCRITURA DE LA ADVERTENCIA ===
            f.write("--- ANÁLISIS DE NO IDEALIDAD (MODULACIÓN DE CANAL) ---\n")
            if r_d_calculado != float('inf'):
                f.write(
                    f"Resistencia de salida (r_d) calculada a V_GS={v_gs_arr[idx_vgs_max]:.6f}V: {r_d_calculado/1000:.6f} kOhm\n")
                f.write(
                    f"Parámetro de modulación (λ): {lambda_val:.6f} V^-1\n")
                f.write(
                    "> ADVERTENCIA TÉCNICA: Las curvas de salida muestran una pendiente notable en la zona de saturación.\n")
                f.write(
                    "> Esto indica que el transistor no es una fuente de corriente perfecta (Efecto Early / Modulación de longitud de canal).\n")
                f.write(
                    "> Los cálculos de polarización teórica de este informe utilizan el modelo ideal de Shockley (r_d = infinito).\n")
                f.write("> JUSTIFICACIÓN: Se esperan desviaciones entre estos valores teóricos y los medidos en la protoboard debido a esta fuga resistiva.\n\n")
            else:
                f.write(
                    "Las curvas presentan una planitud ideal (r_d tiende a infinito).\n")
        print(
            f"-> Archivo de datos guardado con éxito en: {ruta_archivo_salida}")
        print("="*50 + "\n")

    with open(ruta_archivo_salida, "a", encoding="utf-8") as f:
        f.write("\n\n=== RESULTADOS TEÓRICOS DE POLARIZACIÓN (GUÍA LAB 3) ===\n")
        f.write(f"Voltaje de alimentación V_DD asumido: {V_DD_LAB} V\n")

        for config in CIRCUITOS_PRUEBA:
            resultados_q = resolver_polarizacion_teorica(
                a_ne,
                b_ne,
                V_DD_LAB,
                config)

            texto_res = f"\n>> CONFIGURACIÓN: {config['nombre']}\n"
            if resultados_q is None:
                texto_res += "  ¡ERROR: El circuito no converge o el transistor entra en corte profundo!\n"
            else:
                texto_res += f"  Punto Q: V_GSQ = {resultados_q['V_GS']:.6f} V,  I_DQ = {resultados_q['I_D'] * 1000:.6f} mA,  V_DSQ = {resultados_q['V_DS']:.6f} V\n"
                texto_res += f"  Voltajes Nodos: V_D = {resultados_q['V_D']:.6f} V,  V_G = {resultados_q['V_G']:.6f} V,  V_S = {resultados_q['V_S']:.6f} V\n"
                texto_res += f"  Corrientes:     I_D = {resultados_q['I_D'] * 1000:.6f} mA,  I_S = {resultados_q['I_S'] * 1000:.6f} mA,  I_G = {resultados_q['I_G']} A\n"

            print(texto_res, end="")
            f.write(texto_res)
