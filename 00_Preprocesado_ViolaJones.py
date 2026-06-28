import cv2
import os
import glob

# =========================================================================
# FASE 0: PIPELINE DE PRE-PROCESADO HOLÍSTICO Y EXTRACCIÓN DE ROI
# =========================================================================
# Autor: Andoni Cabrera Fernández
# Descripción: Procesamiento masivo de 111 GB de vídeo. Implementa
#              submuestreo temporal (1 FPS), detección de Viola-Jones 
#              con padding del 20%, y tolerancia a fallos (Checkpointing).
# =========================================================================

ruta_base_dataset = r"D:\TFG_Fatiga_Andoni\3_Datasets\UTA-RLDD\Videos_Originales\UTA Real-Life Drowsiness Dataset"
ruta_salida_base = r"D:\TFG_Fatiga_Andoni\3_Datasets\Caras_Recortadas_Procesadas"

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
detector_caras = cv2.CascadeClassifier(cascade_path)

total_videos_procesados = 0
total_caras_guardadas = 0

print("Iniciando Pipeline de Pre-procesado UTA-RLDD (1 FPS)...\n")

carpetas_fold = sorted([f for f in os.listdir(ruta_base_dataset) if f.startswith("Fold")])

for fold in carpetas_fold:
    ruta_fold = os.path.join(ruta_base_dataset, fold)
    carpetas_sujetos = sorted([s for s in os.listdir(ruta_fold) if os.path.isdir(os.path.join(ruta_fold, s))])
    
    for sujeto in carpetas_sujetos:
        ruta_sujeto_destino = os.path.join(ruta_salida_base, fold, f"Sujeto_{sujeto}")
        
        # =================================================================
        # MÓDULO DE TOLERANCIA A FALLOS (CHECKPOINTING)
        # Previene la sobreescritura y permite reanudar procesos masivos.
        # =================================================================
        if os.path.exists(ruta_sujeto_destino):
            print(f"[INFO] Omitiendo {fold} | Sujeto {sujeto} -> Caché existente.")
            continue
            
        ruta_sujeto_origen = os.path.join(ruta_fold, sujeto)
        videos = glob.glob(os.path.join(ruta_sujeto_origen, "*.*"))
        videos = [v for v in videos if v.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        for ruta_video in videos:
            nombre_video = os.path.basename(ruta_video) 
            clase_fatiga = nombre_video.split('.')[0]   
            
            carpeta_salida_final = os.path.join(ruta_sujeto_destino, f"Clase_{clase_fatiga}")
            os.makedirs(carpeta_salida_final, exist_ok=True)
                
            print(f"Procesando: {fold} | Sujeto {sujeto} | Video {nombre_video} ...")
            
            # =================================================================
            # EXTRACCIÓN TEMPORAL Y DETECCIÓN FACIAL
            # =================================================================
            cap = cv2.VideoCapture(ruta_video)
            fps_original = cap.get(cv2.CAP_PROP_FPS)
            
            if fps_original <= 0: 
                fps_original = 30 
                
            frame_count = 0
            segundos_procesados = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: 
                    break 
                
                frame_count += 1
                
                # Submuestreo a 1 FPS
                if frame_count % int(fps_original) == 0:
                    segundos_procesados += 1
                    img_limpia = frame.copy()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    caras = detector_caras.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100)
                    )
                    
                    if len(caras) > 0:
                        if len(caras) > 1:
                            caras = sorted(caras, key=lambda c: c[2] * c[3], reverse=True)
                        x, y, w, h = caras[0]
                            
                        # Padding geométrico (20%)
                        pad_w = int(w * 0.20)
                        pad_h = int(h * 0.20)
                            
                        x1 = max(0, x - pad_w)
                        y1 = max(0, y - pad_h)
                        x2 = min(frame.shape[1], x + w + pad_w)
                        y2 = min(frame.shape[0], y + h + pad_h)
                            
                        cara_recortada = img_limpia[y1:y2, x1:x2]
                            
                        if cara_recortada.size > 0:
                            cara_final = cv2.resize(cara_recortada, (224, 224))
                            nombre_archivo = os.path.join(carpeta_salida_final, f"seg_{segundos_procesados}.jpg")
                            cv2.imwrite(nombre_archivo, cara_final)
                            total_caras_guardadas += 1

            cap.release()
            total_videos_procesados += 1

print("\n==========================================")
print("¡PIPELINE DE PRE-PROCESADO FINALIZADO!")
print(f"Vídeos analizados en esta sesión: {total_videos_procesados}")
print(f"Caras extraídas en esta sesión:   {total_caras_guardadas}")
print("==========================================")