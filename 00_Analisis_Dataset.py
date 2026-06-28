"""
00_Analisis_Dataset.py

Script para la auditoría y análisis estadístico del dataset UTA-RLDD.
Recorre la estructura de directorios generada tras el preprocesamiento,
cuantifica las muestras válidas por cada estado de fatiga y genera
una gráfica de balanceo de clases para la memoria técnica.
"""

import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================================
# CONFIGURACIÓN DEL ENTORNO
# =========================================================================
# Completar con la ruta local donde se encuentran las carpetas por Fold
ruta_salida_base = r"" 
nombre_grafico = "descripcion_dataset.png"

clase_0 = 0
clase_5 = 0
clase_10 = 0

print("Iniciando auditoría de muestras extraídas...")

# =========================================================================
# EXTRACCIÓN DE MÉTRICAS POBLACIONALES
# =========================================================================
# Iteramos sobre la jerarquía del dataset: Fold -> Sujeto -> Clase
if ruta_salida_base:
    for fold in os.listdir(ruta_salida_base):
        ruta_fold = os.path.join(ruta_salida_base, fold)
        
        if os.path.isdir(ruta_fold):
            for sujeto in os.listdir(ruta_fold):
                ruta_sujeto = os.path.join(ruta_fold, sujeto)
                
                if os.path.isdir(ruta_sujeto):
                    clase_0 += len(glob.glob(os.path.join(ruta_sujeto, "Clase_0", "*.jpg")))
                    clase_5 += len(glob.glob(os.path.join(ruta_sujeto, "Clase_5", "*.jpg")))
                    clase_10 += len(glob.glob(os.path.join(ruta_sujeto, "Clase_10", "*.jpg")))

total = clase_0 + clase_5 + clase_10

# =========================================================================
# REPORTE EN CONSOLA
# =========================================================================
print("\n" + "="*45)
print("REPORTE DE AUDITORÍA DEL DATASET")
print("="*45)
print(f"Total Clase 0 (Alerta base):         {clase_0}")
print(f"Total Clase 5 (Baja Vigilancia):     {clase_5}")
print(f"Total Clase 10 (Somnolencia):        {clase_10}")
print("-" * 45)
print(f"TOTAL DE IMÁGENES VÁLIDAS:           {total}")
print("=" * 45 + "\n")

# =========================================================================
# RENDERIZADO DEL DIAGRAMA DE DISTRIBUCIÓN
# =========================================================================
if total > 0:
    print(f"Generando gráfico de balanceo de clases: {nombre_grafico}...")

    clases = ['Clase 0\n(Alerta)', 'Clase 5\n(Baja Vigilancia)', 'Clase 10\n(Somnolencia)']
    cantidades = [clase_0, clase_5, clase_10]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    ax = sns.barplot(x=clases, y=cantidades, palette="viridis", hue=clases, legend=False)

    # Inserción de etiquetas numéricas de soporte sobre las barras
    for i, v in enumerate(cantidades):
        ax.text(i, v + (total * 0.01), str(v), ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.title('Distribución de Muestras por Nivel de Fatiga', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Número de Imágenes', fontsize=12, fontweight='bold')
    plt.xlabel('Estado del Conductor', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(nombre_grafico, dpi=300)
    
    print("Proceso finalizado correctamente.")
else:
    print("Aviso: No se han detectado imágenes. Verifica la variable 'ruta_salida_base'.")
