import copy

import openpifpaf
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
import pandas as pd
import os

class Person:
  def __init__(self, id, use_right_hand=True, sitting = True):
    self.user_id = id
    self.use_right_hand = use_right_hand
    self.sitting = sitting

    self.interaction = 'No interaction'
    self.time_at_acquisition = None
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

class interaction:
  def __init__(self, start_time, collection_time, max_samples = 3):
    self.start_time = 0
    self.start_collect = 0
    self.start_rest = 0
    self.start_end = 0
    self.collection_time = collection_time
    self.phase = 'init message'
    self.times = []
    self.data = []
    self.samples = 0
    self.total_repeats = 1
    self.max_samples = max_samples
    self.dist = 1000
    self.dist_old = 1000
    self.velocity = 0
    self.t_old = time.time()

  def reset(self):
      self.start_time = 0
      self.start_collect = 0
      self.start_rest = 0
      self.start_end = 0
      self.phase = 'collect data'
      self.times = []
      self.data = []
      #self.samples = 0  # do not reset number of samples
      self.total_repeats = 1
      self.repeat_num = 0
      self.dist = 1000
      self.dist_old = 1000
      self.velocity = 0
      self.t_old = time.time()
  def annotate_img(self, img, text, color, position = (20, 100)):
    return cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5,
                        cv2.LINE_AA)

  def display_interaction(self, img, person, position):
      return self.annotate_img(img, person.interaction, (0, 0, 255), position)

  def init_message(self, img):
    msg_duration = 3
    count_down_duration = 2
    if (time.time() - self.start_time) < msg_duration:
      img = self.annotate_img(img, self.message_init , (0, 0, 255))

    elif time.time()-self.start_time < msg_duration+count_down_duration:
      img = self.annotate_img(img, str(int(msg_duration + count_down_duration - (time.time() - self.start_time - 1))), (0, 0, 255))

    else:
        self.phase = 'collect data'
        self.start_collect = time.time()
    return img

  def collect_message(self, img, records):
    msg_duration = self.collection_time
    elapsed_time = time.time() - self.start_collect
    if elapsed_time < msg_duration:
      img = self.annotate_img(img, self.message_collect +str(1+int(msg_duration -elapsed_time))+'s', (0, 0, 255), position=(20, 100))

    else:
       self.phase = 'rest'
       self.start_rest = time.time()
       self.samples += 1
       if self.samples >= self.max_samples:
           self.phase = 'end'
           self.start_end = time.time()

    return img, records


  def collect_message_count(self, img, records):
    total_repeats = self.total_repeats
    repeat_num = self.repeat_num
    if repeat_num < total_repeats:
        img = self.annotate_img(img, self.message_collect + str(int(total_repeats - repeat_num))+' times',
                                (0, 0, 255), position = (20, 100))
    else:
       self.phase = 'rest'
       self.repeat_num = 0
       self.dist_old = 9999
       self.dist = 9999
       self.start_rest = time.time()
       self.samples += 1
       if self.samples >= self.max_samples:
           self.phase = 'end'
           self.start_end = time.time()

    return img, records

  def rest_routine(self, frame):
      if (time.time() - self.start_rest < 3.5):
          frame = self.annotate_img(frame, 'Rest', (0, 255, 0), position = (20, 100))
      else:
          self.phase = 'collect data'
          self.start_collect = time.time()

      return frame

  def check_cdt(self, person, person_old, frame, records):
      raise NotImplementedError

  def is_part_detected(self, part1_xy, part1_c, part2_xy, part2_c):
    if len(part1_xy[np.where(part1_c > 0)[0], :])*len(part2_xy[np.where(part2_c > 0)[0], :]) == 0:
        return False
    else:
        return True


  def compute_distance(self, part1_xy, part1_c, part2_xy, part2_c, old=False):

      if self.is_part_detected(part1_xy, part1_c, part2_xy, part2_c):

        if old:
            self.dist_old = np.min(
                cdist(part1_xy[np.where(part1_c > 0)[0], :],
                    part2_xy[np.where(part2_c > 0)[0], :]))
        else:
            self.dist = np.min(
                  cdist(part1_xy[np.where(part1_c > 0)[0], :],
                        part2_xy[np.where(part2_c > 0)[0], :]))



class eye_touching(interaction):
    def __init__(self, start_time, distance_thresh=30, max_samples=5):
        super().__init__(start_time, collection_time=3, max_samples=max_samples)
        self.name = 'Eye touching'
        self.message_init = 'Touch your eye in: '
        self.message_collect = 'Touch your eye '
        self.distance_thresh = distance_thresh
        self.repeat_num = 0

    def check_cdt(self, person, person_old, frame, records):

        if person.use_right_hand:
          self.compute_distance(person.right_hand_xy, person.right_hand_c, person.eyes_xy, person.eyes_c)
          self.compute_distance(person_old.right_hand_xy, person_old.right_hand_c, person_old.eyes_xy, person_old.eyes_c, old=True)
        else:
            self.compute_distance(person.left_hand_xy, person.left_hand_c, person.eyes_xy, person.eyes_c)
            self.compute_distance(person_old.left_hand_xy, person_old.left_hand_c, person_old.eyes_xy, person_old.eyes_c,old=True)

        print('Distance :', self.dist)
        if self.dist < self.distance_thresh:
            if self.dist_old >= self.distance_thresh:
                self.start_collect = time.time()
            person.interaction = self.name
        else:
            if person.interaction == self.name: # Check if previous state was Eye touching
                records = records.append(
                    {'user_id': person.user_id, 'class': self.name, 'sample_nb': self.samples,
                     'start_time': self.start_collect, 'end_time': time.time(),
                     'date_time': time.asctime(time.localtime(time.time())), 'right_hand': int(person.use_right_hand), 'sitting': int(person.sitting)},
                    ignore_index=True)
                self.repeat_num += 1
            person.interaction = 'No interaction'

        frame, records = self.collect_message_count(frame, records)
        return frame, records


class eye_rubbing_light(interaction):
    def __init__(self, start_time, distance_thresh=30, max_samples = 10):
        super().__init__(start_time, collection_time=4, max_samples=max_samples)
        self.name = 'Eye rubbing light'
        self.message_init = 'Rub your eye in: '
        self.message_collect = 'Rub eye (Light) '
        self.distance_thresh = distance_thresh
        self.repeat_num = 0

    def check_cdt(self, person, person_old, frame, records):

        if person.use_right_hand:
          self.compute_distance(person.right_hand_xy, person.right_hand_c, person.eyes_xy, person.eyes_c)
          self.compute_distance(person_old.right_hand_xy, person_old.right_hand_c, person_old.eyes_xy, person_old.eyes_c, old=True)
        else:
            self.compute_distance(person.left_hand_xy, person.left_hand_c, person.eyes_xy, person.eyes_c)
            self.compute_distance(person_old.left_hand_xy, person_old.left_hand_c, person_old.eyes_xy, person_old.eyes_c,old=True)

        print('Distance :', self.dist)
        if self.dist < self.distance_thresh:
            if self.dist_old >= self.distance_thresh:
                self.start_collect = time.time()
            person.interaction = self.name
        else:
            if person.interaction == self.name: # Check if previous state was Eye touching
                records = records.append(
                    {'user_id': person.user_id, 'class': self.name, 'sample_nb': self.samples,
                     'start_time': self.start_collect, 'end_time': time.time(),
                     'date_time': time.asctime(time.localtime(time.time())), 'right_hand': int(person.use_right_hand), 'sitting': int(person.sitting)},
                    ignore_index=True)
                self.repeat_num += 1
            person.interaction = 'No interaction'

        frame, records = self.collect_message_count(frame, records)
        return frame, records

class eye_rubbing_moderate(interaction):
    def __init__(self, start_time, distance_thresh = 40, velocity_thresh = 0.05, max_samples=9):
        super().__init__(start_time, collection_time=6, max_samples=max_samples)
        self.name = 'Eye rubbing moderate'
        self.message_init = 'Rub your eye in: '
        self.message_collect = 'Rub eye (Moderate) '
        self.distance_thresh = distance_thresh
        self.velocity_thresh = velocity_thresh
        self.repeat_num = 0
        self.valid_velocity = False
        self.velcount = time.time()*2

    def check_cdt(self, person, person_old, frame, records):

        if person.use_right_hand:
            intersection_RH_kpts = np.intersect1d(np.where(person_old.right_hand_c > 0)[0],
                                                  np.where(person.right_hand_c > 0)[0]).astype(int)
            previous_RH_pose = person_old.right_hand_xy[intersection_RH_kpts, :]
            new_RH_pose = person.right_hand_xy[intersection_RH_kpts, :]

            if (len(new_RH_pose) * len(previous_RH_pose) > 0):
                velocities = np.sqrt(np.sum((new_RH_pose - previous_RH_pose) ** 2, axis=1)) / (time.time() - self.t_old)
                self.t_old = time.time()
                self.velocity = np.mean(np.mean(velocities))
        else:
            intersection_LH_kpts = np.intersect1d(np.where(person_old.left_hand_c > 0)[0],
                                                  np.where(person.left_hand_c > 0)[0]).astype(int)
            previous_LH_pose = person_old.left_hand_xy[intersection_LH_kpts, :]
            new_LH_pose = person.left_hand_xy[intersection_LH_kpts, :]

            if (len(new_LH_pose) * len(previous_LH_pose) > 0):
                velocities = np.sqrt(np.sum((new_LH_pose - previous_LH_pose) ** 2, axis=1)) / (time.time() - self.t_old)
                self.t_old = time.time()
                self.velocity = np.mean(np.mean(velocities))


        if person.use_right_hand:
          self.compute_distance(person.right_hand_xy, person.right_hand_c, person.eyes_xy, person.eyes_c)
          self.compute_distance(person_old.right_hand_xy, person_old.right_hand_c, person_old.eyes_xy, person_old.eyes_c, old=True)
        else:
            self.compute_distance(person.left_hand_xy, person.left_hand_c, person.eyes_xy, person.eyes_c)
            self.compute_distance(person_old.left_hand_xy, person_old.left_hand_c, person_old.eyes_xy, person_old.eyes_c,old=True)
        print('Distance :', self.dist)
        print('Velocity :', self.velocity)

        if self.dist < self.distance_thresh:
            person.interaction = self.name
            if self.dist_old >= self.distance_thresh:
                self.start_collect = time.time()
                self.velcount = time.time()
            if self.velocity < self.velocity_thresh:
                self.velcount = time.time()
            if time.time() - self.velcount >= 1:
                self.valid_velocity = True

        else:
            if person.interaction == self.name: # Check if previous state was Eye touching
                if self.valid_velocity:
                    records = records.append(
                        {'user_id': person.user_id, 'class': self.name, 'sample_nb': self.samples,
                         'start_time': self.start_collect, 'end_time': time.time(),
                         'date_time': time.asctime(time.localtime(time.time())), 'right_hand': int(person.use_right_hand), 'sitting': int(person.sitting)},
                        ignore_index=True)
                    self.repeat_num += 1
                    self.valid_velocity = False
            person.interaction = 'No interaction'
        print(self.valid_velocity)
        frame, records = self.collect_message_count(frame, records)

        return frame, records

class teeth_brushing(interaction):
    def __init__(self, start_time, distance_thresh=60, velocity_thresh=0.1, max_samples=8):
        super().__init__(start_time, collection_time=6, max_samples=max_samples)
        self.name = 'Teeth brushing'
        self.message_init = 'Brush your teeth in: '
        self.message_collect = 'Brush your teeth '
        self.distance_thresh = distance_thresh
        self.velocity_thresh = velocity_thresh
        self.repeat_num = 0
        self.valid_velocity = False
        self.velcount = time.time()*2

    def check_cdt(self, person, person_old, frame, records):

        if person.use_right_hand:
            intersection_RH_kpts = np.intersect1d(np.where(person_old.right_hand_c > 0)[0],
                                                  np.where(person.right_hand_c > 0)[0]).astype(int)
            previous_RH_pose = person_old.right_hand_xy[intersection_RH_kpts, :]
            new_RH_pose = person.right_hand_xy[intersection_RH_kpts, :]

            if (len(new_RH_pose) * len(previous_RH_pose) > 0):
                velocities = np.sqrt(np.sum((new_RH_pose - previous_RH_pose) ** 2, axis=1)) / (time.time() - self.t_old)
                self.t_old = time.time()
                self.velocity = np.mean(np.mean(velocities))
        else:
            intersection_LH_kpts = np.intersect1d(np.where(person_old.left_hand_c > 0)[0],
                                                  np.where(person.left_hand_c > 0)[0]).astype(int)
            previous_LH_pose = person_old.left_hand_xy[intersection_LH_kpts, :]
            new_LH_pose = person.left_hand_xy[intersection_LH_kpts, :]

            if (len(new_LH_pose) * len(previous_LH_pose) > 0):
                velocities = np.sqrt(np.sum((new_LH_pose - previous_LH_pose) ** 2, axis=1)) / (time.time() - self.t_old)
                self.t_old = time.time()
                self.velocity = np.mean(np.mean(velocities))


        if person.use_right_hand:
          self.compute_distance(person.right_hand_xy, person.right_hand_c, person.eyes_xy, person.eyes_c)
          self.compute_distance(person_old.right_hand_xy, person_old.right_hand_c, person_old.eyes_xy, person_old.eyes_c, old=True)
        else:
            self.compute_distance(person.left_hand_xy, person.left_hand_c, person.eyes_xy, person.eyes_c)
            self.compute_distance(person_old.left_hand_xy, person_old.left_hand_c, person_old.eyes_xy, person_old.eyes_c,old=True)

        print('Distance :', self.dist)
        print('Velocity :', self.velocity)
        if self.dist < self.distance_thresh:
            person.interaction = self.name
            if self.dist_old >= self.distance_thresh:
                self.start_collect = time.time()
                self.velcount = time.time()
            if self.velocity < self.velocity_thresh:
                self.velcount = time.time()
            if time.time() - self.velcount >= 1:
                self.valid_velocity = True

        else:
            if person.interaction == self.name: # Check if previous state was Eye touching
                if self.valid_velocity:
                    records = records.append(
                        {'user_id': person.user_id, 'class': self.name, 'sample_nb': self.samples,
                         'start_time': self.start_collect, 'end_time': time.time(),
                         'date_time': time.asctime(time.localtime(time.time())), 'right_hand': int(person.use_right_hand), 'sitting': int(person.sitting)},
                        ignore_index=True)
                    self.repeat_num += 1
                    self.valid_velocity = False
            person.interaction = 'No interaction'
        print(self.valid_velocity)
        frame, records = self.collect_message_count(frame, records)

        return frame, records

class glasses_readjusting(interaction):
    def __init__(self, start_time, distance_thresh=30, max_samples=5):
        super().__init__(start_time, collection_time=2, max_samples=max_samples)
        self.name = 'Glasses readjusting'
        self.message_init = 'Readjust your glasses in: '
        self.message_collect = 'Readjust glasses '
        self.distance_thresh = distance_thresh
        self.repeat_num = 0

    def check_cdt(self, person, person_old, frame, records):

        # Actual position
        if  self.is_part_detected(person.left_ear_xy,person.left_ear_c, person.right_ear_xy, person.right_ear_c):
            if person.use_right_hand:
                if self.is_part_detected(person.right_hand_xy, person.right_hand_c, person.eyes_xy,
                                      person.eyes_c):

                    glasses_zone = np.concatenate((person.eyes_xy[np.where(person.eyes_c > 0)[0], :],
                                                   person.left_ear_xy[np.where(person.right_ear_c > 0)[0], :],
                                                   person.right_ear_xy[np.where(person.left_ear_c > 0)[0], :]))
                    self.dist = np.min(
                        cdist(glasses_zone,
                              person.right_hand_xy[np.where(person.right_hand_c > 0)[0], :]))


            else:
                if self.is_part_detected(person.left_hand_xy, person.left_hand_c, person.eyes_xy,
                                         person.eyes_c):

                    glasses_zone = np.concatenate((person.eyes_xy[np.where(person.eyes_c > 0)[0], :],
                                                   person.left_ear_xy[np.where(person.right_ear_c > 0)[0], :],
                                                   person.right_ear_xy[np.where(person.left_ear_c > 0)[0], :]))
                    self.dist = np.min(
                        cdist(glasses_zone,
                              person.left_hand_xy[np.where(person.left_hand_c > 0)[0], :]))
        print('Distance new: ', self.dist)

        # Previous position
        if self.is_part_detected(person_old.left_ear_xy, person_old.left_ear_c, person_old.right_ear_xy, person_old.right_ear_c):
            if person.use_right_hand:
                if self.is_part_detected(person_old.right_hand_xy, person_old.right_hand_c, person_old.eyes_xy,
                                         person_old.eyes_c):
                    glasses_zone = np.concatenate((person_old.eyes_xy[np.where(person_old.eyes_c > 0)[0], :],
                                                   person_old.left_ear_xy[np.where(person_old.right_ear_c > 0)[0], :],
                                                   person_old.right_ear_xy[np.where(person_old.left_ear_c > 0)[0], :]))
                    self.dist_old = np.min(
                        cdist(glasses_zone,
                              person_old.right_hand_xy[np.where(person_old.right_hand_c > 0)[0], :]))


            else:
                if self.is_part_detected(person_old.left_hand_xy, person_old.left_hand_c, person_old.eyes_xy,
                                         person_old.eyes_c):
                    glasses_zone = np.concatenate((person_old.eyes_xy[np.where(person_old.eyes_c > 0)[0], :],
                                                   person_old.left_ear_xy[np.where(person_old.right_ear_c > 0)[0], :],
                                                   person_old.right_ear_xy[np.where(person_old.left_ear_c > 0)[0], :]))
                    self.dist_old = np.min(
                        cdist(glasses_zone,
                              person_old.left_hand_xy[np.where(person_old.left_hand_c > 0)[0], :]))

        print('Distance :', self.dist)

        if self.dist < self.distance_thresh:
            if self.dist_old >= self.distance_thresh:
                self.start_collect = time.time()
            person.interaction = self.name
        else:
            if person.interaction == self.name: # Check if previous state was Eye touching
                records = records.append(
                    {'user_id': person.user_id, 'class': self.name, 'sample_nb': self.samples,
                     'start_time': self.start_collect, 'end_time': time.time(),
                     'date_time': time.asctime(time.localtime(time.time())), 'right_hand': int(person.use_right_hand), 'sitting': int(person.sitting)},
                    ignore_index=True)
                self.repeat_num += 1
            person.interaction = 'No interaction'

        frame, records = self.collect_message_count(frame, records)
        return frame, records

class hair_combing(interaction):
    def __init__(self, start_time, distance_thresh=60, max_samples = 5):
        super().__init__(start_time, collection_time=10, max_samples=max_samples)
        self.name = 'Hair combing'
        self.message_init = 'Comb your hair in: '
        self.message_collect = 'Comb your hair '
        self.distance_thresh = distance_thresh
        self.repeat_num = 0

    def check_cdt(self, person, person_old, frame, records):

        if person.use_right_hand:
            if len(np.where(person.right_hand_c > 0)[0]) * len(
                    np.where(person.left_ear_c > 0)[0]) * len(
                np.where(person.right_ear_c > 0)[0]) > 0:
                ears_xy = np.concatenate((person.left_ear_xy[np.where(person.left_ear_c > 0)[0], :],
                                          person.right_ear_xy[np.where(person.left_ear_c > 0)[0], :]))
                self.dist = np.min(
                    cdist(ears_xy,
                          person.right_hand_xy[np.where(person.right_hand_c > 0)[0], :]))

            if len(np.where(person_old.right_hand_c > 0)[0]) * len(
                    np.where(person_old.left_ear_c > 0)[0]) * len(
                np.where(person_old.right_ear_c > 0)[0]) > 0:
                ears_xy = np.concatenate((person_old.left_ear_xy[np.where(person_old.left_ear_c > 0)[0], :],
                                          person_old.right_ear_xy[np.where(person_old.left_ear_c > 0)[0], :]))
                self.dist_old = np.min(
                    cdist(ears_xy,
                          person_old.right_hand_xy[np.where(person_old.right_hand_c > 0)[0], :]))
        else:
            if len(np.where(person.left_hand_c > 0)[0]) * len(
                    np.where(person.left_ear_c > 0)[0]) * len(
                np.where(person.right_ear_c > 0)[0]) > 0:
                ears_xy = np.concatenate((person.left_ear_xy[np.where(person.left_ear_c > 0)[0], :],
                                          person.right_ear_xy[np.where(person.left_ear_c > 0)[0], :]))
                self.dist = np.min(
                    cdist(ears_xy,
                          person.left_hand_xy[np.where(person.left_hand_c > 0)[0], :]))

            if len(np.where(person_old.left_hand_c > 0)[0]) * len(
                    np.where(person_old.left_ear_c > 0)[0]) * len(
                np.where(person_old.right_ear_c > 0)[0]) > 0:
                ears_xy = np.concatenate((person_old.left_ear_xy[np.where(person_old.left_ear_c > 0)[0], :],
                                          person_old.right_ear_xy[np.where(person_old.left_ear_c > 0)[0], :]))
                self.dist_old = np.min(
                    cdist(ears_xy,
                          person_old.left_hand_xy[np.where(person_old.left_hand_c > 0)[0], :]))

        print('Distance: ', self.dist)

        if self.dist < self.distance_thresh:
            if self.dist_old >= self.distance_thresh:
                self.start_collect = time.time()
            person.interaction = self.name
        else:
            if person.interaction == self.name: # Check if previous state was Eye touching
                records = records.append(
                    {'user_id': person.user_id, 'class': self.name, 'sample_nb': self.samples,
                     'start_time': self.start_collect, 'end_time': time.time(),
                     'date_time': time.asctime(time.localtime(time.time())), 'right_hand': int(person.use_right_hand), 'sitting': int(person.sitting)},
                    ignore_index=True)
                self.repeat_num += 1
            person.interaction = 'No interaction'

        frame, records = self.collect_message_count(frame, records)
        return frame, records


class skin_scratching(interaction):
    def __init__(self, start_time, distance_thresh=80, max_samples = 10):
        super().__init__(start_time, collection_time=4, max_samples=max_samples)
        self.name = 'Skin scratching'
        self.message_init = 'Scratch your face in: '
        self.message_collect = 'Scratch your face '
        self.distance_thresh = distance_thresh
        self.repeat_num = 0

    def check_cdt(self, person, person_old, frame, records):

        if person.use_right_hand:
            self.compute_distance(person.right_hand_xy, person.right_hand_c, person.nose_xy, person.nose_c)
            self.compute_distance(person_old.right_hand_xy, person_old.right_hand_c, person_old.nose_xy,
                                  person_old.nose_c, old=True)

        else:
            self.compute_distance(person.left_hand_xy, person.left_hand_c, person.nose_xy, person.nose_c)
            self.compute_distance(person_old.left_hand_xy, person_old.left_hand_c, person_old.nose_xy,
                                  person_old.nose_c, old=True)
        print('Distance: ', self.dist)


        if self.dist < self.distance_thresh:
            if self.dist_old >= self.distance_thresh:
                self.start_collect = time.time()
            person.interaction = self.name
        else:
            if person.interaction == self.name:
                records = records.append(
                    {'user_id': person.user_id, 'class': self.name, 'sample_nb': self.samples,
                     'start_time': self.start_collect, 'end_time': time.time(),
                     'date_time': time.asctime(time.localtime(time.time())), 'right_hand': int(person.use_right_hand), 'sitting': int(person.sitting)},
                    ignore_index=True)
                self.repeat_num += 1
            person.interaction = 'No interaction'

        frame, records = self.collect_message_count(frame, records)
        return frame, records

class eating(interaction):
    def __init__(self, start_time, distance_thresh=80, max_samples = 3):
        super().__init__(start_time, collection_time=8, max_samples=max_samples)
        self.name = 'Eating'
        self.message_init = 'Start eating in: '
        self.message_collect = 'Eat '
        self.distance_thresh = distance_thresh
        self.repeat_num = 0

    def check_cdt(self, person, person_old, frame, records):

        if person.use_right_hand:
            if len(person.right_hand_xy[np.where(person.right_hand_c > 0)[0], :]) * len(
                  person.nose_xy[np.where(person.nose_c > 0)[0], :]) > 0:
              self.dist = np.min(
                  cdist(person.nose_xy[np.where(person.nose_c > 0)[0], :],
                        person.right_hand_xy[np.where(person.right_hand_c > 0)[0], :]))
              print('Distance: ', self.dist)

            if len(person_old.right_hand_xy[np.where(person_old.right_hand_c > 0)[0], :]) * len(
                  person_old.nose_xy[np.where(person_old.nose_c > 0)[0], :]) > 0:
              self.dist_old = np.min(
                  cdist(person_old.nose_xy[np.where(person_old.nose_c > 0)[0], :],
                        person_old.right_hand_xy[np.where(person_old.right_hand_c > 0)[0], :]))

        else:
            if len(person.left_hand_xy[np.where(person.left_hand_c > 0)[0], :]) * len(
                    person.nose_xy[np.where(person.nose_c > 0)[0], :]) > 0:
                self.dist = np.min(
                    cdist(person.nose_xy[np.where(person.nose_c > 0)[0], :],
                          person.left_hand_xy[np.where(person.left_hand_c > 0)[0], :]))
            if len(person_old.left_hand_xy[np.where(person_old.left_hand_c > 0)[0], :]) * len(
                    person_old.nose_xy[np.where(person_old.nose_c > 0)[0], :]) > 0:
                self.dist_old = np.min(
                    cdist(person_old.nose_xy[np.where(person_old.nose_c > 0)[0], :],
                          person_old.left_hand_xy[np.where(person_old.left_hand_c > 0)[0], :]))
                print('Distance: ', self.dist)

        if self.dist < self.distance_thresh:
            if self.dist_old >= self.distance_thresh:
                self.start_collect = time.time()
            person.interaction = self.name
        else:
            if person.interaction == self.name: # Check if previous state was Eye touching
                records = records.append(
                    {'user_id': person.user_id, 'class': self.name, 'sample_nb': self.samples,
                     'start_time': self.start_collect, 'end_time': time.time(),
                     'date_time': time.asctime(time.localtime(time.time())), 'right_hand': int(person.use_right_hand), 'sitting': int(person.sitting)},
                    ignore_index=True)
                self.repeat_num += 1
            person.interaction = 'No interaction'

        frame, records = self.collect_message_count(frame, records)
        return frame, records

class make_up_application(interaction):
    def __init__(self, start_time, distance_thresh=80, max_samples = 5):
        super().__init__(start_time, collection_time=10, max_samples=max_samples)
        self.name = 'Make up application'
        self.message_init = 'Apply make up in: '
        self.message_collect = 'Apply make up '
        self.distance_thresh = distance_thresh
        self.repeat_num = 0

    def check_cdt(self, person, person_old, frame, records):

        if person.use_right_hand:
            self.compute_distance(person.right_hand_xy, person.right_hand_c, person.nose_xy, person.nose_c)
            self.compute_distance(person_old.right_hand_xy, person_old.right_hand_c, person_old.nose_xy, person.nose_c, old=True)
        else:
            self.compute_distance(person.left_hand_xy, person.left_hand_c, person.nose_xy, person.nose_c)
            self.compute_distance(person_old.left_hand_xy, person_old.left_hand_c, person_old.nose_xy, person_old.nose_c, old=True)
        print('Distance: ', self.dist)

        if self.dist < self.distance_thresh:
            if self.dist_old >= self.distance_thresh:
                self.start_collect = time.time()
            person.interaction = self.name
        else:
            if person.interaction == self.name: # Check if previous state was Eye touching
                records = records.append(
                    {'user_id': person.user_id, 'class': self.name, 'sample_nb': self.samples,
                     'start_time': self.start_collect, 'end_time': time.time(),
                     'date_time': time.asctime(time.localtime(time.time())), 'right_hand': int(person.use_right_hand), 'sitting': int(person.sitting)},
                    ignore_index=True)
                self.repeat_num += 1
            person.interaction = 'No interaction'

        frame, records = self.collect_message_count(frame, records)
        return frame, records

class make_up_removal(interaction):
    def __init__(self, start_time, distance_thresh=80, max_samples = 5):
        super().__init__(start_time, collection_time=10, max_samples=max_samples)
        self.name = 'Make up removal'
        self.message_init = 'Remove make up in: '
        self.message_collect = 'Remove make up '
        self.distance_thresh = distance_thresh
        self.repeat_num = 0

    def check_cdt(self, person, person_old, frame, records):

        if person.use_right_hand:
            self.compute_distance(person.right_hand_xy, person.right_hand_c, person.nose_xy, person.nose_c)
            self.compute_distance(person_old.right_hand_xy, person_old.right_hand_c, person_old.nose_xy, person.nose_c, old=True)
        else:
            self.compute_distance(person.left_hand_xy, person.left_hand_c, person.nose_xy, person.nose_c)
            self.compute_distance(person_old.left_hand_xy, person_old.left_hand_c, person_old.nose_xy, person_old.nose_c, old=True)
        print('Distance: ', self.dist)

        if self.dist < self.distance_thresh:
            if self.dist_old >= self.distance_thresh:
                self.start_collect = time.time()
            person.interaction = self.name
        else:
            if person.interaction == self.name: # Check if previous state was Eye touching
                records = records.append(
                    {'user_id': person.user_id, 'class': self.name, 'sample_nb': self.samples,
                     'start_time': self.start_collect, 'end_time': time.time(),
                     'date_time': time.asctime(time.localtime(time.time())), 'right_hand': int(person.use_right_hand), 'sitting': int(person.sitting)},
                    ignore_index=True)
                self.repeat_num += 1
            person.interaction = 'No interaction'

        frame, records = self.collect_message_count(frame, records)
        return frame, records


def distance(a,b):
  return np.sqrt(np.sum((a-b)**2))