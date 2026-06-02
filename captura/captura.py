import serial
import csv
import time
import os

transistor = "2N5457"
muestras_esperadas = 5000

try:
    ser = serial.Serial('COM9', 921600, timeout=2)
except serial.SerialException as e:
    print(f"Error al abrir el puerto COM3: {e}")
    exit(1)

time.sleep(2)
ser.reset_input_buffer()

# 2. Solicitar y leer V_GS
print("Solicitando lectura de V_GS al ESP32...")
ser.write(b"V_GS\n")
v_gs_str = ser.readline().decode('utf-8', errors='ignore').strip()

try:
    v_gs = float(v_gs_str)
    print(f"-> Valor V_GS detectado: {v_gs:.4f} V")
except ValueError:
    print(f"Error: No se pudo interpretar V_GS. Se recibió: '{v_gs_str}'")
    ser.close()
    exit(1)

carpeta_destino = f"./transistores/{transistor}"
os.makedirs(carpeta_destino, exist_ok=True)
nombre_archivo = f"{carpeta_destino}/{v_gs:.4f}V.CSV"

print(f"Iniciando captura de {muestras_esperadas} muestras...")
ser.write(b"CAPTURAR\n")

with open(nombre_archivo, mode='w', newline='') as archivo_csv:
    escritor = csv.writer(archivo_csv)

    muestras_recibidas = 0
    while muestras_recibidas < muestras_esperadas:
        linea = ser.readline().decode('utf-8', errors='ignore').strip()

        if linea:
            datos = linea.split(',')
            if len(datos) == 3:
                escritor.writerow(datos)
                muestras_recibidas += 1
                if muestras_recibidas % 500 == 0 or muestras_recibidas == muestras_esperadas:
                    print(f"Progreso: {muestras_recibidas}/{muestras_esperadas}")

ser.close()
print(f"\n¡Éxito! Captura finalizada y guardada en: {nombre_archivo}")