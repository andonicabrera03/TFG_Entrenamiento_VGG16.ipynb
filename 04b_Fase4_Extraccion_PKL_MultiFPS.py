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
# FASE 4b: EXTRACCIÓN OFFLINE A PRUEBA DE FALLOS (CORREGIDO RGB)
# =========================================================================

# 1. Configuración de Rutas
ruta_base_dataset = r"D:\TFG_Fatiga_Andoni\3_Datasets\UTA-RLDD\Videos_Originales\UTA Real-Life Drowsiness Dataset"
ruta_modelos_tflite = r"D:\TFG_Fatiga_Andoni\Modelos_Fase4"
ruta_salida_base = r"D:\TFG_Fatiga_Andoni\Resultados_PKL"
ruta_checkpoints = os.path.join(ruta_salida_base, "Checkpoints_Temporales")

os.makedirs(ruta_salida_base, exist_ok=True)
os.makedirs(ruta_checkpoints, exist_ok=True)

# 2. Inicialización de los Intérpretes TFLite
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

print("Intérpretes TFLite listos para inferencia.\n")

# 3. Inicialización de MediaPipe Face Landmarker
ruta_mp_task = 'face_landmarker.task'
if not os.path.exists(ruta_mp_task):
    print("Descargando modelo de MediaPipe (face_landmarker.task)...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, ruta_mp_task)

base_options = python.BaseOptions(model_asset_path=ruta_mp_task)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector_landmarks = vision.FaceLandmarker.create_from_options(options)

# 4. Funciones Auxiliares de Inferencia (Renombramos parámetro a img_rgb por claridad)
def predecir_tflite(interprete, input_details, output_details, tensor_input):
    interprete.set_tensor(input_details['index'], tensor_input)
    interprete.invoke()
    return interprete.get_tensor(output_details['index'])[0]

def extraer_roi_preprocesada(img_rgb, landmarks, indices, w, h, margen_x, margen_y):
    x_coords = [int(landmarks[i].x * w) for i in indices]
    y_coords = [int(landmarks[i].y * h) for i in indices]

    x_min, x_max = max(0, min(x_coords) - margen_x), min(w, max(x_coords) + margen_x)
    y_min, y_max = max(0, min(y_coords) - margen_y), min(h, max(y_coords) + margen_y)

    recorte = img_rgb[y_min:y_max, x_min:x_max]
    if recorte.size == 0: return None

    recorte_res = cv2.resize(recorte, (224, 224))
    # Normalización MobileNetV2: de [0, 255] a [-1.0, 1.0]
    recorte_res = (recorte_res.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(recorte_res, axis=0)

# 5. Pipeline de Extracción Principal con CHECKPOINTS
total_videos_procesados = 0
total_frames_analizados = 0

print("Iniciando Pipeline Robusto de Extracción con TFLite...\n")

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
            
            identificador_unico = f"{fold}_{sujeto}_{nombre_video}"
            ruta_checkpoint = os.path.join(ruta_checkpoints, f"{identificador_unico}.pkl")

            if os.path.exists(ruta_checkpoint):
                print(f"Saltando (ya procesado): {identificador_unico}")
                continue

            clase_fatiga = nombre_video.split('.')[0]
            try: clase_real = int(clase_fatiga)
            except ValueError: clase_real = -1
                
            print(f"Procesando: {fold} | Sujeto {sujeto} | Video {nombre_video}")
            
            datos_video_actual = []
            
            cap = cv2.VideoCapture(ruta_video)
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Conversión a RGB
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape
                
                # MediaPipe usa la imagen RGB
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                resultados = detector_landmarks.detect(mp_image)

                if resultados.face_landmarks:
                    landmarks = resultados.face_landmarks[0]
                    
                    # CORRECCIÓN VITAL: Pasamos img_rgb en lugar de frame (BGR) a la IA
                    t_ojos = extraer_roi_preprocesada(img_rgb, landmarks, [33, 133, 362, 263, 70, 300], w, h, 20, 30)
                    t_boca = extraer_roi_preprocesada(img_rgb, landmarks, [78, 308, 13, 14, 0, 17], w, h, 20, 20)

                    if t_ojos is not None and t_boca is not None:
                        p_ojos = predecir_tflite(interpreter_ojos, input_details_ojos, output_details_ojos, t_ojos)
                        p_boca = predecir_tflite(interpreter_boca, input_details_boca, output_details_boca, t_boca)

                        dato_frame = {
                            'video': f"{sujeto}_{nombre_video}",
                            'clase_real': clase_real,
                            'frame_idx': frame_idx,
                            'prob_ojo_cerrado': float(p_ojos[0]),
                            'prob_bostezo': float(p_boca[0])
                        }

                        datos_video_actual.append(dato_frame)
                        total_frames_analizados += 1

                frame_idx += 1
            
            cap.release()
            total_videos_procesados += 1

            with open(ruta_checkpoint, 'wb') as f:
                pickle.dump(datos_video_actual, f)

# =========================================================================
# 6. FUSIÓN FINAL (Consolidación de Checkpoints)
# =========================================================================
print("\n" + "="*40)
print("Fase de Extracción completada. Consolidando archivos temporales...")

datos_totales_30fps = []
datos_totales_15fps = []
datos_totales_5fps = []
datos_totales_1fps = []

archivos_temporales = glob.glob(os.path.join(ruta_checkpoints, "*.pkl"))

for archivo in tqdm(archivos_temporales, desc="Consolidando datos"):
    with open(archivo, 'rb') as f:
        datos_video = pickle.load(f)
        
        for dato in datos_video:
            datos_totales_30fps.append(dato)
            if dato['frame_idx'] % 2 == 0: datos_totales_15fps.append(dato)
            if dato['frame_idx'] % 6 == 0: datos_totales_5fps.append(dato)
            if dato['frame_idx'] % 30 == 0: datos_totales_1fps.append(dato)

print("\nGuardando matrices de inferencia consolidadas...")
with open(os.path.join(ruta_salida_base, 'datos_TFLITE_30fps.pkl'), 'wb') as f: pickle.dump(datos_totales_30fps, f)
with open(os.path.join(ruta_salida_base, 'datos_TFLITE_15fps.pkl'), 'wb') as f: pickle.dump(datos_totales_15fps, f)
with open(os.path.join(ruta_salida_base, 'datos_TFLITE_5fps.pkl'), 'wb') as f: pickle.dump(datos_totales_5fps, f)
with open(os.path.join(ruta_salida_base, 'datos_TFLITE_1fps.pkl'), 'wb') as f: pickle.dump(datos_totales_1fps, f) 

print("\n" + "="*40)
print("¡EXTRACCIÓN TFLITE COMPLETADA CON ÉXITO!")
print(f"Vídeos: {total_videos_procesados} | Frames TFLite: {total_frames_analizados}")
print("="*40)