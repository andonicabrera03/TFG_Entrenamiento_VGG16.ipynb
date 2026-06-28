import cv2
import os
import glob
import math
import numpy as np
import pickle
import urllib.request
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================================
# FASE 5a: EXTRACCIÓN TOPOLÓGICA 3D OFFLINE (EAR y MAR) -> Multi-FPS
# =========================================================================
# Autor: Andoni Cabrera Fernández
# Descripción: Inferencia geométrica explícita (TinyML en CPU). Abandona 
#              la inferencia de píxeles (Fase 4) por distancias espaciales 
#              euclidianas en 3D (Invarianza Rotacional).
#              Serializa EAR y MAR crudos en archivos PKL.
# =========================================================================

# 1. Configuración de Rutas (Estructura idéntica a las fases anteriores)
ruta_base_dataset = r"D:\TFG_Fatiga_Andoni\3_Datasets\UTA-RLDD\Videos_Originales\UTA Real-Life Drowsiness Dataset"
# Podemos guardar los PKL geométricos en una carpeta separada para no pisar los de la red neuronal
ruta_salida_base = r"D:\TFG_Fatiga_Andoni\Resultados_Geometria_PKL"

os.makedirs(ruta_salida_base, exist_ok=True)

# 2. Inicialización de MediaPipe Face Landmarker (Local / CPU)
ruta_mp_task = 'face_landmarker.task'
if not os.path.exists(ruta_mp_task):
    print("Descargando modelo de MediaPipe (face_landmarker.task)...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, ruta_mp_task)

print("Iniciando Motor Geométrico MediaPipe en CPU...")
base_options = python.BaseOptions(model_asset_path=ruta_mp_task)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False, # Desactivado para maximizar rendimiento
    output_facial_transformation_matrixes=False
)
detector_landmarks = vision.FaceLandmarker.create_from_options(options)
print("Motor matemático inicializado correctamente.\n")

# 3. Módulo Matemático: Distancia Euclidiana 3D
def distancia_puntos_3d(p1, p2, w, h):
    """Calcula la distancia Euclidiana en 3D para garantizar Invarianza Rotacional"""
    x1, y1, z1 = p1.x * w, p1.y * h, p1.z * w
    x2, y2, z2 = p2.x * w, p2.y * h, p2.z * w
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def calcular_ear_3d(landmarks, w, h):
    """Eye Aspect Ratio (EAR) tridimensional (6 puntos por ojo)"""
    izq = [33, 160, 158, 133, 153, 144]
    der = [362, 385, 387, 263, 373, 380]

    def ear_ojo(p):
        v1 = distancia_puntos_3d(landmarks[p[1]], landmarks[p[5]], w, h)
        v2 = distancia_puntos_3d(landmarks[p[2]], landmarks[p[4]], w, h)
        horiz = distancia_puntos_3d(landmarks[p[0]], landmarks[p[3]], w, h)
        return (v1 + v2) / (2.0 * horiz) if horiz > 0 else 0

    return (ear_ojo(izq) + ear_ojo(der)) / 2.0

def calcular_mar_3d(landmarks, w, h):
    """Mouth Aspect Ratio (MAR) tridimensional"""
    p_sup, p_inf = 13, 14
    p_izq, p_der = 78, 308

    vertical = distancia_puntos_3d(landmarks[p_sup], landmarks[p_inf], w, h)
    horizontal = distancia_puntos_3d(landmarks[p_izq], landmarks[p_der], w, h)
    return vertical / horizontal if horizontal > 0 else 0

# 4. Pipeline de Extracción Principal
datos_30fps = []
datos_15fps = []
datos_5fps = []

total_videos_procesados = 0
total_frames_analizados = 0

print("Iniciando Pipeline de Extracción Topológica 3D (Multi-FPS)...\n")

carpetas_fold = sorted([f for f in os.listdir(ruta_base_dataset) if f.startswith("Fold")])

for fold in carpetas_fold:
    ruta_fold = os.path.join(ruta_base_dataset, fold)
    carpetas_sujetos = sorted([s for s in os.listdir(ruta_fold) if os.path.isdir(os.path.join(ruta_fold, s))])
    
    for sujeto in carpetas_sujetos:
        ruta_sujeto_origen = os.path.join(ruta_fold, sujeto)
        videos = glob.glob(os.path.join(ruta_sujeto_origen, "*.*"))
        videos = [v for v in videos if v.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        for ruta_video in videos:
            nombre_video = os.path.basename(ruta_video) 
            clase_fatiga = nombre_video.split('.')[0]   
            
            try:
                clase_real = int(clase_fatiga)
            except ValueError:
                clase_real = -1
                
            print(f"Procesando Geometría: {fold} | Sujeto {sujeto} | Video {nombre_video} ...")
            
            # =================================================================
            # INFERENCIA GEOMÉTRICA ESPACIAL
            # =================================================================
            cap = cv2.VideoCapture(ruta_video)
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: 
                    break 
                
                # Inferencia a 30 FPS (Se procesa cada frame)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                resultados = detector_landmarks.detect(mp_image)

                if resultados.face_landmarks:
                    landmarks = resultados.face_landmarks[0]
                    
                    ear_actual = calcular_ear_3d(landmarks, w, h)
                    mar_actual = calcular_mar_3d(landmarks, w, h)

                    # Diccionario de coordenadas del fotograma actual
                    dato_frame = {
                        'video': f"{sujeto}_{nombre_video}", # ID único para trazabilidad
                        'clase_real': clase_real, 
                        'frame_idx': frame_idx,
                        'ear': ear_actual,
                        'mar': mar_actual
                    }

                    # =====================================================
                    # MULTIPLEXACIÓN TEMPORAL (30, 15 y 5 FPS)
                    # =====================================================
                    datos_30fps.append(dato_frame)               
                    if frame_idx % 2 == 0: datos_15fps.append(dato_frame) 
                    if frame_idx % 6 == 0: datos_5fps.append(dato_frame)  
                    
                    total_frames_analizados += 1

                frame_idx += 1
            
            cap.release()
            total_videos_procesados += 1

# =================================================================
# SERIALIZACIÓN FINAL EN ARCHIVOS PICKLE
# =================================================================
print("\nSerializando coordenadas trigonométricas en archivos PKL...")
with open(os.path.join(ruta_salida_base, 'geometria_30fps.pkl'), 'wb') as f: pickle.dump(datos_30fps, f)
with open(os.path.join(ruta_salida_base, 'geometria_15fps.pkl'), 'wb') as f: pickle.dump(datos_15fps, f)
with open(os.path.join(ruta_salida_base, 'geometria_5fps.pkl'), 'wb') as f: pickle.dump(datos_5fps, f)

print("\n==========================================")
print("¡PIPELINE DE EXTRACCIÓN GEOMÉTRICA FINALIZADO!")
print(f"Vídeos analizados en esta sesión: {total_videos_procesados}")
print(f"Fotogramas calculados (Euclides): {total_frames_analizados}")
print(f"Archivos PKL generados en:        {ruta_salida_base}")
print("==========================================")