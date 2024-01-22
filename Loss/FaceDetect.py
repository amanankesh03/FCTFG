import face_alignment
import numpy as np
from PIL import Image
import cv2
import imageio
import torch
from torchvision.io import read_video
from torchvision import transforms
import pickle
import matplotlib.pyplot as plt
from landmarks import PIPNet

def face_detect_all(face_detector_fa, frame, thresh=0.97, use_only_det=True, visualize=False):

    if max(frame.shape[0], frame.shape[1]) > 640:
        print('Rescaling and detecting at 640 resolution')
        scale_factor = max(frame.shape[0], frame.shape[1]) / 640.0
        resized_frame = cv2.resize(frame, (int(frame.shape[1] // scale_factor), int(frame.shape[0] // scale_factor)))
    else:
        scale_factor = 1
        resized_frame = frame[..., :3]
    bboxes = face_detector_fa.face_detector.detect_from_image(resized_frame[..., (2, 1, 0)])
    all_landmarks = face_detector_fa.get_landmarks(resized_frame[..., (2, 1, 0)], detected_faces=bboxes)
    
   
    if len(bboxes) == 0:
        return [], []
    results = np.array(bboxes)[:, :-1] * scale_factor
    all_landmarks = np.array(all_landmarks) * scale_factor
    best_results, best_landmarks = [], []
    width_thresh = 20
    for ind, (rect, landmarks) in enumerate(zip(results, all_landmarks)):
        conf = bboxes[ind][-1]
        if conf > thresh:
            if use_only_det:
                x1, y1, x2, y2 = int(max(rect[0], 0)), int(max(0, rect[1])), \
                                 int(min(rect[2], frame.shape[1])), int(min(rect[3], frame.shape[0]))
            else:
                face_left, face_right = min(landmarks[:, 0]), max(landmarks[:, 0])
                eye_left, eye_right = landmarks[19], landmarks[24]
                eye_mid = (eye_left + eye_right) // 2
                nose = landmarks[33]
                chin = max(landmarks[:, 1])
                lip_mid = (landmarks[52] + landmarks[66]) / 2
                x1, y1, x2, y2 = int(max(0, min(rect[0], face_left))), int(max(0, min(rect[1], eye_mid[1] - (nose[1] - eye_mid[1]) // 2))), \
                                 int(min(frame.shape[1], max(rect[2], face_right))), int(min(frame.shape[0], max(rect[3], chin)))
                if visualize:
                    new_frame = frame.copy()
                    cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    for lm in landmarks:
                        cv2.circle(new_frame, (int(lm[0]), int(lm[1])), 1, (0, 255, 255))
                    cv2.imshow('detection', new_frame)
                    cv2.waitKey(0)
        
            width = x2 - x1
            if width < width_thresh:
                continue
            if y2 <= y1 or x2 <= x1:
                continue
            best_landmarks.append(landmarks)
            best_results.append([x1, y1, x2, y2])
    if len(best_results) > 0:
        return np.array(best_results), np.array(all_landmarks), resized_frame
    else:
        return [], [], []

def get_ref_img(lms, frames):
    # print(frames.shape)
    lms = np.array(lms)
    lip_landmarks = lms[:, 76:93, :]

    # Find the difference between the highest and lowest y-coordinate for each frame's lip landmarks
    ref_idx = np.argmax(np.max(lip_landmarks[:, :, 1], axis=1) - np.min(lip_landmarks[:, :, 1], axis=1))
    ref_img = frames[ref_idx] / 255
    shape_y, shape_x = ref_img.shape[1] / 512, ref_img.shape[2] / 512
    
    if not isinstance(lms, list):
        lms = lms.tolist()

    lms = lms[ref_idx][3:30] + lms[ref_idx][55:59]
    lms = [[int(x / shape_x), int(y / shape_y)] for x, y in lms]
    lms = np.array(lms).astype(np.int32).reshape((-1, 1, 2))

    mask = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(mask, [lms], 255)
    mask = torch.from_numpy(mask / 255).float()

    ref_img = torch.from_numpy(ref_img)

    # plt.imshow(ref_img)
    # plt.show()
    ref_img = ref_img.permute(2, 0, 1)
    ref_img = resize(ref_img)


    # print(ref_img.shape)

    mask = torch.stack([mask, mask, mask], dim=0)

    print(ref_img.shape, mask.shape)

    plt.imshow(mask.permute(1,2,0) * 255)
    plt.show()
    ref_img = (ref_img * mask).float()

    return ref_img

def get_cropped_frame(frame, face_box, extend=0.2):
        frame = frame[0]
        face_box = face_box[0]
        _, height, width = frame.shape
        # print('face box : ', face_box)
        x1, y1, x2, y2 = face_box[0][0], face_box[0][1], face_box[0][2], face_box[0][3]

        print(x1, y1, x2, y2)
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        x1 -= int(w * (1. + extend - 1) / 2)
        y1 += int(h * (1. + extend - 1) / 2)
        x2 += int(w * (1. + extend - 1) / 2)
        y2 += int(h * (1. + extend - 1) / 2)

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, width - 1)
        y2 = min(y2, height - 1)


        print(y1, y2, x1, x2, frame.shape)
        # # Crop the faces using PyTorch operations
        cropped = frame[:,y1:y2, x1:x2]
        cropped = transforms.ToPILImage()(cropped).resize((512, 512))

        # plt.imshow(cropped)
        # plt.show()

        cropped = np.array(cropped)
        print('cropped ', cropped.shape)
        return cropped

def get_frames(video_path):
    frames = imageio.get_reader(video_path)
    return frames
    
def landmarks_from_facedetector(frames, face_detector_fa):
    face_boxes = []
    for frame in frames:
        if i > 3:
            break
        frame = np.array(frame)
        face_box, landmarks, resized_frame = face_detect_all(face_detector_fa, frame, use_only_det=True)
        face_boxes.append(face_box)
        i +=1
    return face_boxes

def get_landmarks(frames, face_boxes):

    landmarks_list = []
    resized_frames = []
    cropped_video = []
    for frame, face_box in zip(frames, face_boxes):
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        face_box = torch.from_numpy(face_box).unsqueeze(0)
        cropped = get_cropped_frame(frame, face_box)
        #landmarks
        landmarks = pipnet(frame, face_box)

        landmarks_list.append(landmarks[0])
        cropped_video.append(cropped)

    landmarks_list = np.stack(landmarks_list)
    cropped_video = np.stack(cropped_video)

    return landmarks_list, cropped_video

face_detector_fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
pipnet = PIPNet()

video_path = '/home/amanankesh/working_dir/LIA-working/Motion-with-ref/inference_data/src.mp4' 

rf, lms, crv = ref_from_video(video_path, face_detector_fa, pipnet)
print(rf.shape, lms.shape, crv.shape)

# for frame in crv:
#     print(lms)
#     plt.imshow(frame)
#     plt.show()