from ultralytics import YOLO

old = YOLO("runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/weights/best.engine")
new = YOLO("runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH300-2/weights/best.engine")

for name, model in [("Old", old), ("New", new)]:
    m = model.val(
        data="data.yaml",
        imgsz=640,
        conf=0.1,
    )

    print(name)
    print("Precision:", m.box.mp)
    print("Recall:", m.box.mr)
    print("mAP50:", m.box.map50)
    print("mAP50-95:", m.box.map)