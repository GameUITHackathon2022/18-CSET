import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


from src.utils.torch_utils import select_device
from src.models.experimental import attempt_load
from src.utils.general import check_img_size, non_max_suppression
from src.utils.augmentations import  letterbox

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

class LoadImages1:  # for inference
    def __init__(self, img, img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.img = img
        self.nf = 1
        self.cap = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration

        self.count += 1
        img0 = self.img

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return "", img, img0, self.cap

def load_model(model_path, imgsz = 640):
    device=''
    device = select_device(device)
    half=False
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(model_path,device='cuda')  # load FP32 model
    if half:
        model.half()  # to FP16
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    return model, names, stride, device, half

def predict(model, names, imgsz, stride, device, half, process_img):
    augment=False
    visualize=False
    conf_thres=0.25
    iou_thres=0.45
    max_det=1000
    agnostic_nms=False
    classes=None

    dataset = LoadImages1(process_img, img_size=imgsz, stride=stride)

    return_data = []
    for path, img, im0s, vid_cap in dataset:
        
        img = torch.from_numpy(img).to('cuda')
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred = model(img, augment=augment, visualize=visualize)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det_cpu = det.cpu().tolist()
#                print(det_cpu)
                # Write results
                for *xyxy, conf, cls in reversed(det_cpu):
                    bbox = xyxy
                    conf = float(conf)
                    cls = names[int(cls)]
                    return_data.append({'bbox': bbox, 'score': conf, 'cls': cls})

    return return_data



