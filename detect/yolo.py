from ultralytics import YOLO

model = YOLO("../training/best.pt")




results = model.predict('../input_video/A_3.mp4',save=True, save_dir ='./output')
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)