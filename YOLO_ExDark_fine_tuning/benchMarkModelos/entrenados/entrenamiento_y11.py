from ultralytics import YOLO

model_n11 = YOLO('yolo11n.pt')
resultados11 = model_n11.train(data='../../data/data.yaml', epochs=200, project='resultados_200_epoc', name='yolov11n_200_epoc', imgsz=512, batch=8, patience=10)