"""
04b_Fase4_Extraccion_PKL_MultiFPS.py

Script para la extracción predictiva (offline) a prueba de fallos.
Implementa el paradigma de "Redes Expertas Desacopladas" (Fase 4). 
Utiliza la malla topológica de MediaPipe para aislar los ojos y la boca,
y ejecuta inferencia en CPU utilizando modelos MobileNetV2 cuantizados a 
TFLite (INT8/Float16) para evaluar la oclusión palpebral y los bostezos.

Se procesan y guardan los tensores a 30, 15, 5 y 1 FPS.
Incluye sistema de Checkpoints temporales para guardar automáticamente el 
progreso por vídeo y evitar la pérdida de datos en ejecuciones largas.
"""

import cv2
import os
import glob
import numpy as np
import tensorflow as tf
import pickle
import urllib.request
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================================
# 1. CONFIGURACIÓN DEL ENTORNO Y RUTAS
# =========================================================================
# Completar con las rutas locales antes de la ejecución
ruta_base_dataset = r""
ruta_modelos_tflite = r""
ruta_salida_base = r""
ruta_checkpoints = os.path.join(ruta_salida_base, "Checkpoints_Temporales")

os.makedirs(ruta_checkpoints, exist_ok=True)

# =========================================================================
# 2. CARGA DE MODELOS CUANTIZADOS (TFLITE)
# =========================================================================
def preparar_interprete(nombre_archivo):
    ruta = os.path.join(ruta_modelos_tflite, nombre_archivo)
    print(f"Cargando intérprete: {nombre_archivo}...")
    interprete = tf.lite.Interpreter(model_path=ruta)
    interprete.allocate_tensors()
    return interprete

interpreter_ojos = preparar_interprete('experto_ojos_cuantizado.tflite')
interpreter_boca = preparar_interprete('experto_boca_cuantizado.tflite')

input_details_ojos = interpreter_ojos.get_input_details()[0]
output_details_ojos = interpreter_ojos.get_output_details()[0]

input_details_boca = interpreter_boca.get_input_details()[0]
output_details_boca = interpreter_boca.get_output_details()[0]

print("Intérpretes TFLite cargados y listos para inferencia en CPU.\n")

# =========================================================================
# 3. INICIALIZACIÓN DE LA MALLA TOPOLÓGICA (MEDIAPIPE)
# =========================================================================
ruta_mp_task = 'face_landmarker.task'

# Descarga automática del modelo base si no existe
if not os.path.exists(ruta_mp_task):
    print("Descargando modelo base de MediaPipe (face_landmarker.task)...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, ruta_mp_task)
    print("Descarga completada.\n")

base_options = python.BaseOptions(model_asset_path=ruta_mp_task)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector_malla = vision.FaceLandmarker.create_from_options(options)

# Índices topológicos estándar para extracción (Ojos y Boca)
INDICES_OJO_IZQ = [33, 160, 158, 133, 153, 144]
INDICES_OJO_DER = [362, 385, 387, 263, 373, 380]
INDICES_BOCA = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# =========================================================================
# 4. FUNCIONES DE EXTRACCIÓN E INFERENCIA TFLITE
# =========================================================================

def predecir_tflite(interprete, input_details, output_details, tensor_input):
    """
    Ejecuta el grafo computacional de TFLite con el tensor de entrada
    y devuelve la probabilidad predictiva (Logit).
    """
    interprete.set_tensor(input_details['index'], tensor_input)
    interprete.invoke()
    return interprete.get_tensor(output_details['index'])[0]


def extraer_roi_preprocesada(img_rgb, landmarks, indices, w, h, margen_x, margen_y):
    """
    Aísla una sub-región geométrica a partir de las coordenadas de los landmarks,
    aplica un margen de seguridad dinámico y normaliza la matriz para la inferencia.
    """
    x_coords = [int(landmarks[i].x * w) for i in indices]
    y_coords = [int(landmarks[i].y * h) for i in indices]
    
    x_min, x_max = max(0, min(x_coords) - margen_x), min(w, max(x_coords) + margen_x)
    y_min, y_max = max(0, min(y_coords) - margen_y), min(h, max(y_coords) + margen_y)
    
    roi = img_rgb[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return None
        
    roi_resized = cv2.resize(roi, (224, 224))
    
    # Pre-procesamiento análogo a MobileNetV2 (Normalización [-1, 1])
    roi_float = roi_resized.astype(np.float32)
    roi_preprocesada = (roi_float / 127.5) - 1.0
    
    return np.expand_dims(roi_preprocesada, axis=0)

# =========================================================================
# 5. PIPELINE DE EXTRACCIÓN PRINCIPAL (CHECKPOINTING Y MULTI-FPS)
# =========================================================================
total_videos_procesados = 0
total_frames_analizados = 0

print("Iniciando Pipeline Robusto de Extracción Multimodal con TFLite...\n")

if ruta_base_dataset:
    carpetas_fold = sorted([f for f in os.listdir(ruta_base_dataset) if f.startswith("Fold")])
    
    for fold in carpetas_fold:
        ruta_fold = os.path.join(ruta_base_dataset, fold)
        carpetas_sujetos = sorted([s for s in os.listdir(ruta_fold) if os.path.isdir(os.path.join(ruta_fold, s))])
        
        for sujeto in carpetas_sujetos:
            ruta_sujeto = os.path.join(ruta_fold, sujeto)
            videos = glob.glob(os.path.join(ruta_sujeto, "*.*"))
            videos = [v for v in videos if v.lower().endswith(('.mp4', '.avi', '.mov'))]
            
            for ruta_video in videos:
                nombre_video = os.path.basename(ruta_video)
                clase_real = int(nombre_video.split('.')[0])
                
                # Checkpointing: Ignoramos vídeos ya procesados exitosamente
                ruta_ckpt = os.path.join(ruta_checkpoints, f"{fold}_{sujeto}_{nombre_video}_multi_fps.pkl")
                if os.path.exists(ruta_ckpt):
                    print(f"[SKIP] El vídeo {nombre_video} ya fue procesado. Cargando desde caché...")
                    continue
                
                print(f"-> Procesando: {fold} | Sujeto {sujeto} | Video {nombre_video}")
                
                cap = cv2.VideoCapture(ruta_video)
                fps_video = cap.get(cv2.CAP_PROP_FPS)
                if fps_video <= 0: fps_video = 30
                
                # Cálculo de saltos de fotogramas para evaluación Multi-FPS
                salto_15fps = max(1, int(fps_video / 15))
                salto_5fps = max(1, int(fps_video / 5))
                salto_1fps = max(1, int(fps_video / 1))
                
                # Contenedores temporales por frecuencia de muestreo
                frames_30fps, frames_15fps, frames_5fps, frames_1fps = [], [], [], []
                
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if num_frames <= 0: num_frames = 1000 # Estimación segura
                
                frame_count = 0
                
                with tqdm(total=num_frames, desc="Analizando frames", leave=False) as pbar:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        # Corrección cromática al espacio RGB nativo
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                        resultado = detector_malla.detect(mp_image)
                        
                        if resultado.face_landmarks:
                            landmarks = resultado.face_landmarks[0]
                            h, w, _ = frame.shape
                            
                            # Extracción de sub-tensores
                            tensor_ojo_izq = extraer_roi_preprocesada(img_rgb, landmarks, INDICES_OJO_IZQ, w, h, 15, 15)
                            tensor_ojo_der = extraer_roi_preprocesada(img_rgb, landmarks, INDICES_OJO_DER, w, h, 15, 15)
                            tensor_boca = extraer_roi_preprocesada(img_rgb, landmarks, INDICES_BOCA, w, h, 20, 20)
                            
                            # Inferencia en caso de extracción exitosa
                            if tensor_ojo_izq is not None and tensor_ojo_der is not None and tensor_boca is not None:
                                prob_izq = predecir_tflite(interpreter_ojos, input_details_ojos, output_details_ojos, tensor_ojo_izq)
                                prob_der = predecir_tflite(interpreter_ojos, input_details_ojos, output_details_ojos, tensor_ojo_der)
                                prob_boca = predecir_tflite(interpreter_boca, input_details_boca, output_details_boca, tensor_boca)
                                
                                # Consolidación heurística (Peor caso ocular)
                                prob_ojo_cerrado = float(max(prob_izq[1], prob_der[1]))
                                prob_bostezo = float(prob_boca[1])
                                
                                fila = {
                                    'fold': fold,
                                    'sujeto': sujeto,
                                    'video': nombre_video,
                                    'real': clase_real,
                                    'frame': frame_count,
                                    'prob_ojo_cerrado': prob_ojo_cerrado,
                                    'prob_bostezo': prob_bostezo
                                }
                                
                                # Clasificación y almacenamiento temporal por tasa FPS
                                frames_30fps.append(fila)
                                
                                if frame_count % salto_15fps == 0:
                                    frames_15fps.append(fila)
                                    
                                if frame_count % salto_5fps == 0:
                                    frames_5fps.append(fila)

                                if frame_count % salto_1fps == 0:
                                    frames_1fps.append(fila)
                                    
                                total_frames_analizados += 1
                        
                        frame_count += 1
                        pbar.update(1)
                
                cap.release()
                
                # Salvaguarda del estado (Checkpoint persistente)
                diccionario_resultados = {
                    '30fps': frames_30fps,
                    '15fps': frames_15fps,
                    '5fps': frames_5fps,
                    '1fps': frames_1fps
                }
                
                with open(ruta_ckpt, 'wb') as f:
                    pickle.dump(diccionario_resultados, f)
                    
                total_videos_procesados += 1

# =========================================================================
# 6. CONSOLIDACIÓN DE CHECKPOINTS (FUSIÓN FINAL)
# =========================================================================
print("\n" + "="*45)
print("Extracción completada. Consolidando archivos temporales...")

datos_totales_30fps = []
datos_totales_15fps = []
datos_totales_5fps = []
datos_totales_1fps = []

if ruta_salida_base:
    archivos_ckpt = glob.glob(os.path.join(ruta_checkpoints, "*.pkl"))

    for archivo in tqdm(archivos_ckpt, desc="Consolidando datos"):
        with open(archivo, 'rb') as f:
            datos_video = pickle.load(f)
            datos_totales_30fps.extend(datos_video['30fps'])
            datos_totales_15fps.extend(datos_video['15fps'])
            datos_totales_5fps.extend(datos_video['5fps'])
            datos_totales_1fps.extend(datos_video['1fps'])

    print("\nGuardando matrices de inferencia consolidadas...")
    with open(os.path.join(ruta_salida_base, 'datos_TFLITE_30fps.pkl'), 'wb') as f:
        pickle.dump(datos_totales_30fps, f)
    with open(os.path.join(ruta_salida_base, 'datos_TFLITE_15fps.pkl'), 'wb') as f:
        pickle.dump(datos_totales_15fps, f)
    with open(os.path.join(ruta_salida_base, 'datos_TFLITE_5fps.pkl'), 'wb') as f:
        pickle.dump(datos_totales_5fps, f)
    with open(os.path.join(ruta_salida_base, 'datos_TFLITE_1fps.pkl'), 'wb') as f:
        pickle.dump(datos_totales_1fps, f)

    print("="*45)
    print("¡PROCESO MULTIMODAL FINALIZADO CON ÉXITO!")
    print(f"Vídeos totales extraídos: {total_videos_procesados}")
    print(f"Tensores inferidos:       {total_frames_analizados}")
    print("="*45)
