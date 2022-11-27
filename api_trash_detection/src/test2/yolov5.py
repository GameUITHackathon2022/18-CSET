import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from thuann_utils import LoadImages

def load_model(weights, device, imgsz):
    model = attempt_load(weights, device=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    names = model.module.names if hasattr(model, "module") else model.names
    colors = []
    return model, names, colors, stride, imgsz

def predict(model, process_img):
    conf_thres = 0.5
    iou_thres = 0.5
    classes = None
    agnostic_nms = False
    
    label_map = ["back_cccd", "back_cmtnd", "front_cccd", "front_cmtnd"]
    # Functions
    resize = torch.nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
    normalize = lambda x, mean=0.5, std=0.25: (x - mean) / std

    # Image
    im = process_img[::-1]  # HWC, BGR to RGB
    im = np.ascontiguousarray(np.asarray(im).transpose((2, 0, 1)))  # HWC to CHW
    im = torch.tensor(im).unsqueeze(0) / 255.0  # to Tensor, to BCWH, rescale
    im = resize(normalize(im))

    # Inference
    pred = model(im)[1][0]
    print(pred)
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    # print("pred", pred)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            det_cpu = det.cpu().tolist()

            for *xyxy, conf, cls in reversed(det_cpu):
                bbox = xyxy
                conf = float(conf)
                cls = names[int(cls)]

    return label_map[label]

def inference(
    model,
    device,
    names,
    process_img,
    imgsz,
    stride,
    augment,
    conf_thres,
    iou_thres,
    classes=None,
    agnostic_nms=False,
    half=False,
):
    with torch.no_grad():
        dataset = LoadImages(process_img, img_size=imgsz, stride=stride)
        return_data = []
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

            # print("pred", pred)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    det_cpu = det.cpu().tolist()

                    for *xyxy, conf, cls in reversed(det_cpu):
                        bbox = xyxy
                        conf = float(conf)
                        cls = names[int(cls)]

                        return_data.append({"bbox": bbox, "score": conf, "cls": cls})
    return return_data

if __name__=="__main__":
    device = "cpu"
    half = False
    conf_thres = 0.25
    iou_thres = 0.45
    classes = None
    agnostic_nms = True
    augment = True
    imgsz = 640
    thickness = 3

    model_path = "./best.pt"
    img_path = "test.jpg"
    process_img = cv2.imread(img_path)
    model, names, colors, stride, imgsz = load_model(model_path, device, imgsz)
    start_time = time.time()
#    result = inference(model, img)
    result = inference(
        model,
        device,
        names,
        process_img,
        imgsz,
        stride,
        augment,
        conf_thres,
        iou_thres,
        classes=None,
        agnostic_nms=False,
        half=False,
    )
    print(result)
    print(time.time()-start_time)






