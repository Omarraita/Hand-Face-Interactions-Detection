
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

from classes import *

print('Cuda available: ', torch.cuda.is_available())
print('OpenPifPaf version', openpifpaf.__version__)
print('PyTorch version', torch.__version__)
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30-wholebody')
classes = ['eye_touching', 'eye_rubbing_light', 'eye_rubbing_moderate', 'Glasses', 'Hair_combing', 'Skin_scratching', 'Eating', 'Teeth_brushing', 'make_up_application', 'make_up_removal']

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')
cv2.moveWindow('frame', 350, 30)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))


frame_num = 0


# Parameters
user_id = 10
right_handed = True
sitting = True
minutes = 3

# Check if user id already exists


person = Person(user_id, use_right_hand=right_handed, sitting = True)
person_old = copy.deepcopy(person)
# Init classes
interactions = [eye_touching(time.time(), distance_thresh=50, max_samples=1),
                eye_rubbing_light(time.time(), distance_thresh=50, max_samples=1),
                eye_rubbing_moderate(time.time(), distance_thresh=50, velocity_thresh=20, max_samples=1),
                teeth_brushing(time.time(), distance_thresh=65, velocity_thresh=20, max_samples=1),
                glasses_readjusting(time.time(), distance_thresh=50, max_samples=1),
                hair_combing(time.time(), distance_thresh=70, max_samples=1),
                skin_scratching(time.time(), distance_thresh=60, max_samples=1),
                make_up_application(time.time(), distance_thresh=60, max_samples=1),
                make_up_removal(time.time(), distance_thresh=60, max_samples=1),
                eating(time.time(), distance_thresh=70, max_samples=1)]

classes_order = np.random.randint(len(interactions), size=120)

t0 = time.time()

for i in classes_order:

    if time.time()-t0 > 60*minutes:
        break
    # Dataframe
    records = pd.DataFrame(columns=['user_id', 'class', 'sample_nb', 'start_time', 'end_time', 'date_time', 'right_hand', 'sitting'])
    interaction = interactions[i]
    print(classes_order)
    print(i, interaction.name)

    while(True):
        ret, frame = cap.read()

        # Downscale
        scale_percent = 50
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        if frame is None:
            break
        else:
            im = Image.fromarray(frame)
            predictions, gt_anns, image_meta = predictor.pil_image(im)

            if predictions == []:
                continue

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


            # Keypoints detection
            coordinates = predictions[0].data
            # Get specific data
            person.get_person_data(coordinates)
            person.time_at_acquisition = time.time()

            # Finite State Machine
            if interaction.phase == 'init message':
                print(interaction.phase)
                frame = interaction.init_message(frame)

            elif interaction.phase == 'collect data':
                print(interaction.phase)
                frame, records = interaction.check_cdt(person, person_old, frame, records)

            elif interaction.phase == 'rest':
                print(interaction.phase)
                frame = interaction.rest_routine(frame)

            elif interaction.phase == 'end':
                #break
                if time.time()-interaction.start_end < 1.5:
                    frame = interaction.annotate_img(frame, 'Rest', (100, 255, 0))
                else:
                    interaction.reset()
                    break

            # Save person data from previous acquisition
            person_old = copy.deepcopy(person)

            # Display interaction
            frame = interaction.display_interaction(frame, person, (20,900))
            # Save to out video
            #out.write(cv2.resize(frame,(640,480)))

            #print('frame', frame_num)
            frame_num +=1

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if(os.path.exists('records.csv')):
        old_records = pd.read_csv('records.csv')
        records = old_records.append(records)

    records.to_csv('records.csv', index =False)

print('before relase')
cap.release()
#out.release()
cv2.destroyAllWindows()

print(time.time()-t0)
