const int NUM_MUESTRAS = 5000; 
const int NUM_MUESTRAS_V_GS = 100; // Corrección de typo
const int CANTIDAD_PERIODOS = 2;
const float FRECUENCIA_SENAL = 1.0;
const double MICROSEG_IN_SEG = 1000000.0;
const float MILLIVOLTS_IN_VOLT = 1000.0;

const float VCC = 3.3;
const int RESOLUCION_ADC = 4095; // Se cambió 'ADC' para evitar conflictos de palabras reservadas
const float R1 = 4250;
const float R2 = 750;
const float FACTOR_DIVISOR = (R1 + R2) / R2; 

// Pines originales para ESP32 WROOM (Todos pertenecen al ADC1)
const int pinADC1 = 34; // Va  (Pin de solo entrada, ideal para ADC)
const int pinADC2 = 35; // Vb  (Pin de solo entrada, ideal para ADC)
const int pinADC3 = 32; // V_GS (Pin con pull-up/down, excelente para ADC)

// Los arreglos consumen ~60KB de memoria. La ESP32 WROOM tiene 520KB de SRAM, entra perfecto.
unsigned long t_arr[NUM_MUESTRAS];
uint32_t mv1_arr[NUM_MUESTRAS];
uint32_t mv2_arr[NUM_MUESTRAS];

void setup() {
  Serial.begin(921600);
  
  // Forzamos la resolución a 12 bits por seguridad
  analogReadResolution(12); 
}

void loop() {
  if (Serial.available() > 0) {
    String comando = Serial.readStringUntil('\n');
    comando.trim();
    
    if (comando == "CAPTURAR") {
      double periodo_muestreo_us = (MICROSEG_IN_SEG * CANTIDAD_PERIODOS) / (FRECUENCIA_SENAL * NUM_MUESTRAS);
      
      for (int i = 0; i < NUM_MUESTRAS; i++) {        
        unsigned long tiempo_inicio = micros();
        t_arr[i] = tiempo_inicio;
        
        uint32_t lectura_va_1 = analogReadMilliVolts(pinADC1);
        mv2_arr[i] = analogReadMilliVolts(pinADC2);
        uint32_t lectura_va_2 = analogReadMilliVolts(pinADC1);
        
        mv1_arr[i] = (lectura_va_1 + lectura_va_2) / 2;
        
        unsigned long tiempo_gastado = micros() - tiempo_inicio;  
        if (periodo_muestreo_us > tiempo_gastado) {
          delayMicroseconds(periodo_muestreo_us - tiempo_gastado);
        }
      }
      
      for (int i = 0; i < NUM_MUESTRAS; i++) {
        double tiempo = (t_arr[i] - t_arr[0]) / MICROSEG_IN_SEG; 

        float va = FACTOR_DIVISOR * (mv1_arr[i] / MILLIVOLTS_IN_VOLT);
        float vb = FACTOR_DIVISOR * (mv2_arr[i] / MILLIVOLTS_IN_VOLT);
        
        Serial.print(tiempo, 6);
        Serial.print(",");
        Serial.print(va, 6);
        Serial.print(",");
        Serial.println(vb, 6);
      }
    }
    
    else if (comando == "V_GS") {
      uint32_t suma_mv = 0;
      for(int i = 0; i < NUM_MUESTRAS_V_GS; i++){
        suma_mv += analogReadMilliVolts(pinADC3);
        delay(1);
      }
      // Cast a (float) para asegurar precisión decimal en el promedio
      float promedio_mv = (float)suma_mv / NUM_MUESTRAS_V_GS;
      float v_gs = -1.0 * (promedio_mv / MILLIVOLTS_IN_VOLT);
      Serial.println(v_gs, 4);
    }
  }
}