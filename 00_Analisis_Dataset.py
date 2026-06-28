import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
ruta_salida_base = r"D:\TFG_Fatiga_Andoni\3_Datasets\Caras_Recortadas_Procesadas"

# Dónde se va a guardar la imagen (por defecto en la misma carpeta donde ejecutes el script)
nombre_grafico = "descripcion_dataset.png"

clase_0, clase_5, clase_10 = 0, 0, 0

print("Analizando las caras extraídas...")

# ==========================================
# CONTEO DE IMÁGENES
# ==========================================
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

print("\n=== RESULTADOS FINALES PARA OVERLEAF ===")
print(f"Total Clase 0 (Alerta):          {clase_0}")
print(f"Total Clase 5 (Baja Vigilancia): {clase_5}")
print(f"Total Clase 10 (Somnolencia):    {clase_10}")
print(f"----------------------------------------")
print(f"TOTAL DE IMÁGENES VÁLIDAS:       {total}")
print("========================================\n")

# ==========================================
# GENERACIÓN DEL GRÁFICO DE BARRAS
# ==========================================
print("Generando el gráfico de barras...")

clases = ['Clase 0\n(Alerta)', 'Clase 5\n(Baja Vigilancia)', 'Clase 10\n(Somnolencia)']
cantidades = [clase_0, clase_5, clase_10]

# Configurar el estilo del gráfico
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))

# Crear el diagrama de barras
ax = sns.barplot(x=clases, y=cantidades, palette="viridis")

# Añadir los números encima de cada barra
for i, v in enumerate(cantidades):
    ax.text(i, v + (total*0.01), str(v), ha='center', va='bottom', fontweight='bold', fontsize=12)

# Títulos y etiquetas
plt.title('Distribución de Muestras por Nivel de Fatiga', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Número de Imágenes', fontsize=12, fontweight='bold')
plt.xlabel('Estado del Conductor', fontsize=12, fontweight='bold')

# Ajustar márgenes
plt.tight_layout()

# Guardar la imagen
plt.savefig(nombre_grafico, dpi=300)
print(f"Gráfico guardado con éxito como: '{nombre_grafico}'")