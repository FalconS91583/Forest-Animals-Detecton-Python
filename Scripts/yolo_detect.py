import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Definicja i analiza argumentów wejściowych podanych przez użytkownika
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()


# Przetwarzanie argumentów podanych przez użytkownika
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Sprawdzenie czy podana ścieżka do modelu istnieje i jest poprawna
if (not os.path.exists(model_path)):
    print('WARNING: Model path is invalid or model was not found. Using default yolov8s.pt model instead.')
    model_path = 'yolov8s.pt'

# Załadowanie modelu do pamięci oraz uzyskanie mapy etykiet
model = YOLO(model_path, task='detect')
labels = model.names

# Analiza źródła obrazu (plik, folder, wideo, kamera USB)
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Analiza rozdzielczości zadanej przez użytkownika
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Weryfikacja czy nagrywanie jest możliwe oraz inicjalizacja nagrywania
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Inicjalizacja nagrywania
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Ładowanie lub inicjalizacja źródła obrazu
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb': 

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Ustawienie rozdzielczości kamery lub wideo, jeśli zadana przez użytkownika
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

# Ustawienie kolorów dla ramek detekcji (schemat kolorów Tableu 10)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Inicjalizacja zmiennych kontrolnych i statusowych
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Pętla inferencyjna
while True:

    t_start = time.perf_counter()

    # Załadowanie klatki ze źródła obrazu
    if source_type == 'image' or source_type == 'folder': 
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': 
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb': 
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Zmiana rozdzielczości klatki do zadanej przez użytkownika
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Uruchomienie inferencji na załadowanej klatce
    results = model(frame, verbose=False)

    # Ekstrakcja wyników inferencji
    detections = results[0].boxes

    # Inicjalizacja zmiennej do liczenia obiektów w klatce
    object_count = 0

    # Iteracja przez wszystkie wykrycia, uzyskanie współrzędnych ramek, klasy i pewności
    for i in range(len(detections)):

        # Pobranie współrzędnych ramki
        xyxy_tensor = detections[i].xyxy.cpu() 
        xyxy = xyxy_tensor.numpy().squeeze() 
        xmin, ymin, xmax, ymax = xyxy.astype(int) 

        # Pobranie ID klasy oraz nazwy klasy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Pobranie wartości pewności wykrycia
        conf = detections[i].conf.item()

        # Rysowanie ramki, jeśli pewność jest większa niż próg
        if conf > 0.5:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) 
            label_ymin = max(ymin, labelSize[1] + 10) 
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) 
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 

            # Prosty przykład: zliczanie obiektów
            object_count = object_count + 1

    # Obliczanie i wyświetlanie liczby klatek na sekundę (FPS) dla wideo lub kamery
    if source_type == 'video' or source_type == 'usb':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) 
    
    # Wyświetlenie wyników detekcji
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) 
    cv2.imshow('YOLO detection results',frame) 
    if record: recorder.write(frame)

    # Obsługa klawiszy (kontrola programu)
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Naciśnij 'q', aby wyjść
        break
    elif key == ord('s') or key == ord('S'): # Naciśnij 's', aby zapauzować inferencję
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Naciśnij 'p', aby zapisać obraz z wynikami
        cv2.imwrite('capture.png',frame)
    
    # Obliczanie FPS dla aktualnej klatki
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Dodanie FPS do bufora, aby obliczyć średni FPS
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Obliczanie średniego FPS z ostatnich klatek
    avg_frame_rate = np.mean(frame_rate_buffer)


# Czyszczenie zasobów po zakończeniu programu
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
if record: recorder.release()
cv2.destroyAllWindows()
