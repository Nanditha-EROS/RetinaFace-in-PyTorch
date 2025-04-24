
from __future__ import print_function
import os
import cv2
import time
import torch
import numpy as np
import argparse
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import torch.backends.cudnn as cudnn

# ---------------------- CONFIGURATION ----------------------

video_path = "curve/Desi Boyz_04.mp4"  #  GIVE YOUR VIDEO PATH HERE
trained_model_path = "./weights/Resnet50_Final.pth"

# ---------------------- ARGUMENTS ----------------------

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--network', default='resnet50', help='Backbone: mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False)
parser.add_argument('--confidence_threshold', default=0.02, type=float)
parser.add_argument('--top_k', default=5000, type=int)
parser.add_argument('--nms_threshold', default=0.4, type=float)
parser.add_argument('--keep_top_k', default=750, type=int)
parser.add_argument('--vis_thres', default=0.6, type=float)
args = parser.parse_args([])

# ---------------------- MODEL FUNCTIONS ----------------------

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    print('Missing keys:{}'.format(len(model_keys - ckpt_keys)))
    print('Unused checkpoint keys:{}'.format(len(ckpt_keys - model_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0
    return True

def remove_prefix(state_dict, prefix):
    print('Removing prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print(f'Loading pretrained model from {pretrained_path}')
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    pretrained_dict = remove_prefix(pretrained_dict['state_dict'] if "state_dict" in pretrained_dict else pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# ---------------------- MAIN ----------------------

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = cfg_mnet if args.network == "mobile0.25" else cfg_re50
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, trained_model_path, args.cpu)
    net.eval()
    print('Finished loading model!')

    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    output_faces_dir = 'output_faces_video'
    os.makedirs(output_faces_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        original_img = frame.copy()
        annotated_img = frame.copy()

        img = np.float32(frame)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)
        print(f"[Frame {frame_num}] Net forward time: {time.time() - tic:.4f}")

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        boxes = decode(loc.data.squeeze(0), priors.data, cfg['variance']) * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), priors.data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2]] * 5).to(device)
        landms = (landms * scale1).cpu().numpy()

        inds = np.where(scores > args.confidence_threshold)[0]
        boxes, landms, scores = boxes[inds], landms[inds], scores[inds]
        order = scores.argsort()[::-1][:args.top_k]
        boxes, landms, scores = boxes[order], landms[order], scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets, landms = dets[keep, :], landms[keep]
        dets, landms = dets[:args.keep_top_k, :], landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        for idx, b in enumerate(dets):
            if b[4] < args.vis_thres:
                continue

            b_int = list(map(int, b))
            bbox = b_int[:4]
            keypoints = {
                'left_eye': (b_int[4 + 1], b_int[4 + 2]),
                'right_eye': (b_int[4 + 3], b_int[4 + 4]),
                'nose': (b_int[4 + 5], b_int[4 + 6]),
                'mouth_left': (b_int[4 + 7], b_int[4 + 8]),
                'mouth_right': (b_int[4 + 9], b_int[4 + 10]),
            }

            print(f"\n[Frame {frame_num} - Face {idx+1} Keypoints]")
            for key, (x, y) in keypoints.items():
                print(f"{key}: ({x}, {y})")
                cv2.circle(annotated_img, (x, y), 2, (0, 255, 255), 4)

            cv2.rectangle(annotated_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            face_crop = original_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if face_crop.size > 0:
                face_crop_resized = cv2.resize(face_crop, (150, 150))
                face_filename = os.path.join(output_faces_dir, f"frame{frame_num}_face{idx+1}.jpg")
                cv2.imwrite(face_filename, face_crop_resized)

        # Save annotated frame
        annotated_frame_path = os.path.join(output_faces_dir, f"frame{frame_num}_annotated.jpg")
        cv2.imwrite(annotated_frame_path, annotated_img)

    cap.release()
    print("\n✅ Finished processing video.")
