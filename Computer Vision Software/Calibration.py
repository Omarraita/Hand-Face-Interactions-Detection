import copy
import math

import openpifpaf
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import numpy as np


def filter_signal(x, y):
    x = np.array(x)
    y = np.array(y)
    xx = np.linspace(x.min(),x.max(), 1000)
    # interpolate + smooth
    itp = interp1d(x,y, kind='linear')
    window_size, poly_order = 101, 3
    yy_sg = savgol_filter(itp(xx), window_size, poly_order)

    return xx, yy_sg

class Person:
  def __init__(self, id, use_right_hand):
    self.user_id = id
    self.use_right_hand = use_right_hand
    self.interaction = 'No interaction'
    self.time_at_acquisition = None
    # Instruction times
    self.instruction1_time = None
    self.start_aq_time = None
    self.instruction2_time = None
    self.instruction3_time = None
    # Eyes
    self.eyes_keypoints = np.union1d(np.array(range(41,51)),np.array(range(60,72)))
    self.eyes_xy = None
    self.eyes_c = None

    # Left hand
    self.left_hand_keypoints = np.array(range(92,113))
    self.left_hand_xy = []
    self.left_hand_c = []
    self.lh_velocity = 0

    # Right hand
    self.right_hand_keypoints = np.array(range(113,134))
    self.right_hand_xy = []
    self.right_hand_c = []
    self.rh_velocity = 0

    # Mouth
    self.mouth_keypoints = np.array(range(72,88))
    self.mouth_xy = []
    self.mouth_c = []

    # Nose
    self.nose_keypoints = np.array(range(51,60))
    self.nose_xy = []
    self.nose_c = []

    # Left ear
    self.left_ear_keypoints = np.array(range(38,41))
    self.left_ear_xy = []
    self.left_ear_c = []

    # Right ear
    self.right_ear_keypoints = np.array(range(24,27))
    self.right_ear_xy = []
    self.right_ear_c = []

    # Jaw
    self.jaw_keypoints = np.array(range(27,38))
    self.jaw_xy = []
    self.jaw_c = []

  def get_person_data(self, coordinates):

      # Eyes
      self.eyes_xy = coordinates[self.eyes_keypoints - 1][:, 0:2]
      self.eyes_c = coordinates[self.eyes_keypoints - 1][:, 2]

      # Left hand
      self.left_hand_xy = coordinates[self.left_hand_keypoints - 1][:, 0:2]
      self.left_hand_c = coordinates[self.left_hand_keypoints - 1][:, 2]

      # Right hand
      self.right_hand_xy = coordinates[self.right_hand_keypoints - 1][:, 0:2]
      self.right_hand_c = coordinates[self.right_hand_keypoints - 1][:, 2]

      # Mouth
      self.mouth_xy = coordinates[self.mouth_keypoints - 1][:, 0:2]
      self.mouth_c = coordinates[self.mouth_keypoints - 1][:, 2]

      # Nose
      self.nose_xy = coordinates[self.nose_keypoints - 1][:, 0:2]
      self.nose_c = coordinates[self.nose_keypoints - 1][:, 2]

      # Left ear
      self.left_ear_xy = coordinates[self.left_ear_keypoints - 1][:, 0:2]
      self.left_ear_c = coordinates[self.left_ear_keypoints - 1][:, 2]

      # Right ear
      self.right_ear_xy = coordinates[self.right_ear_keypoints - 1][:, 0:2]
      self.right_ear_c = coordinates[self.right_ear_keypoints - 1][:, 2]

      # Jaw
      self.jaw_xy = coordinates[self.jaw_keypoints - 1][:, 0:2]
      self.jaw_c = coordinates[self.jaw_keypoints - 1][:, 2]



def distance(a,b):
  return np.sqrt(np.sum((a-b)**2))


def annotate_img(img, text, color, position=(20, 100)):
    return cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2,
                       cv2.LINE_AA)



print('Cuda available: ', torch.cuda.is_available())
print('OpenPifPaf version', openpifpaf.__version__)
print('PyTorch version', torch.__version__)
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30-wholebody')



cap = cv2.VideoCapture(0)


# Parameters
user_id = 10
right_handed = True

# Check if user id already exists
user_ids = pd.read_csv('records.csv')['user_id'].values
while user_id in user_ids:
    print(user_id)
    user_id += 1

person = Person(user_id, use_right_hand=right_handed)
person_old = copy.deepcopy(person)


# Dataframe
calibration = pd.DataFrame(columns=['user_id','time','wrist_y_data', 'date_time'])

# Init phase
phase = 'instruction1'
person.instruction1_time = time.time()
maximums = [] #[[]]

while(True):
    ret, frame = cap.read()

    # Downscale
    scale_percent = 50
    width = int(frame.shape[1] * scale_percent / 100)
    height0 = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height0)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    if frame is None:
        break
    else:
        im = Image.fromarray(frame)
        predictions, gt_anns, image_meta = predictor.pil_image(im)
        if predictions == []:
            continue
        coordinates = predictions[0].data

        # Get specific data
        person.get_person_data(coordinates)
        person.time_at_acquisition = time.time()

        # Display
        annotation_painter = openpifpaf.show.AnnotationPainter()
        with openpifpaf.show.image_canvas(Image.fromarray(frame)) as ax:
            # fig = plt.figure()
            annotation_painter.annotations(ax, predictions)
            a = plt.axes(ax)
            ax = plt.gca()
            canvas = ax.figure.canvas
            # Force a draw so we can grab the pixel buffer
            canvas.draw()
            # grab the pixel buffer and dump it into a numpy array
            frame = np.array(canvas.renderer.buffer_rgba())

        # Upscale
        scale_percent = 200
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        # Record data

        if person.use_right_hand:
            if person.right_hand_xy[0][1]!=0:
                calibration = calibration.append(
                    {'user_id': person.user_id, 'time': person.time_at_acquisition, 'wrist_y_data': height0-person.right_hand_xy[0][1],
                     'date_time': time.asctime(time.localtime(person.time_at_acquisition)), 'right_hand': person.use_right_hand}, ignore_index=True)
        else:
            if person.left_hand_xy[0][1]!=0:
                calibration = calibration.append(
                    {'user_id': person.user_id, 'time': person.time_at_acquisition, 'wrist_y_data': height0-person.left_hand_xy[0][1],
                     'date_time': time.asctime(time.localtime(person.time_at_acquisition)), 'right_hand': person.use_right_hand}, ignore_index=True)


        if phase == 'instruction1':
            frame = annotate_img(frame, 'Place your wrist in front of the camera and stay still for '+str(math.ceil(10-time.time()+person.instruction1_time))+'s', (0, 255, 0))

            if person.use_right_hand:
                if person.right_hand_xy[0][1]==0:
                    person.instruction1_time = time.time()
                    calibration = pd.DataFrame(columns=['user_id', 'time', 'wrist_y_data', 'date_time'])
            else:
                if person.left_hand_xy[0][1] == 0:
                    person.instruction1_time = time.time()
                    calibration = pd.DataFrame(columns=['user_id', 'time', 'wrist_y_data', 'date_time'])
            if time.time()-person.instruction1_time >= 10:
                phase = 'start_acquisition'

        if phase == 'start_acquisition':
            frame = annotate_img(frame, 'Move your hand up and stay still', (0, 255, 0))
            frame = cv2.line(frame, (0, 480), (1280, 480), (255, 0, 0), thickness=4)
            if len(calibration['wrist_y_data']) > 2:
                if calibration['wrist_y_data'].values[-2] <= height0/2 and calibration['wrist_y_data'].values[-1] > height0/2:
                    person.start_aq_time = time.time()
                    phase = 'up'

        if phase == 'up':
            frame = annotate_img(frame, 'Stay Still for '+str(math.ceil(10-time.time()+person.start_aq_time))+'s', (0, 255, 0))
            frame = cv2.line(frame, (0, int(height/2)), (1280, int(height/2)),
                             (255, 0, 0), thickness=4)
            if time.time()-person.start_aq_time >= 10:
                phase = 'instruction2'
                person.instruction2_time = time.time()

        if phase == 'instruction2':
            frame = annotate_img(frame, 'Move your hand down and stay still', (0, 255, 0))
            frame = cv2.line(frame, (0, int(height/2)), (1280, int(height/2)),
                             (255, 0, 0), thickness=4)
            if calibration['wrist_y_data'].values[-1] <= height0/2:
                person.start_aq_time = time.time()
                phase = 'down'

        if phase == 'down':
            frame = annotate_img(frame, 'Stay Still for ' + str(
                math.ceil(10 - time.time() + person.start_aq_time)) + 's', (0, 255, 0))
            frame = cv2.line(frame, (0, int(height/2)), (1280, int(height/2)),
                             (255, 0, 0), thickness=4)

            if time.time() - person.start_aq_time >= 10:
                phase = 'instruction3'
                person.instruction3_time = time.time()

        if phase == 'instruction3':
            frame = annotate_img(frame, 'Calibration done', (0, 255, 0))
            if time.time() -person.instruction3_time > 2.5:
                break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print(calibration)
calibration.to_csv('calibration_data/calibration'+str(person.user_id)+'.csv', index =False)
cap.release()
cv2.destroyAllWindows()

plt.plot(calibration['time'], calibration['wrist_y_data'])

x, y = filter_signal(calibration['time'], calibration['wrist_y_data'])
plt.plot(x,y)
plt.show()
