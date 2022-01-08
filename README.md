# Hand-Face-Interactions-Detection

The goal of this project is to use Machine learning and Deep learning techniques to classify and detect Human Hand-Face interactions in order to warn users in case of risky hand-face interactions and especiallythe contact with mucosal membranes(e.g., Face skratching, eye-rubbing) to ultimately limit transmission of infectious diseases.  A potential application in ophthalmology: blindness is caused by diseases such as Keratoconus that could be prevented by eye-rubbing cessation. 

# Report
The report of this project can be found [here](https://www.overleaf.com/project/61cb2a3c3b5c5c9c578a1815).

# Requirements

pip3 install openpifpaf==0.12.14 (For computer vision software, v0.12.14 is necessary)

pip3 install torch==1.10.0 (Pytorch Version used for training: compatible with pytorch-lightning)

pip3 install pytorch-lightning  (Practical Pytorch Library that provides built-in modules for training and performance logging)

pip3 install torch==1.10.0 (Pytorch Version used for training: compatible with pytorch-lightning)

# Computer vision software
Uses openpifpaf to detect the user's body. Displays an interaction window with instructions that ask the user to perform specific actions (e.g. eye rubbing, eating , teeth brushing). 

The actions start and end times (unix epochs) are detected and logged into records.csv.  


# Data preparation

- To reproduce the results, clone this repository into your base_folder. Download labeled datasets (ax3, mms+, applewatch) from https://drive.switch.ch/index.php/apps/files/?dir=/Hand-Face%20Interactions&fileid=4315578364 (requires a password) and add them into the Data folder. We expect the directory structure to be the following:

base_folder/Data/  
    codes/  
    Results/ 

# Models 

Link to the trainined LSTM Models https://drive.switch.ch/index.php/apps/files/?dir=/Hand-Face%20Interactions&fileid=4315578364

# Computer Vision codes
- Calibration.py: Asks the user to perform a square signal with his wrists (3 periods of 10 seconds). This will be used to synchronize the times between the computer vision records and the wristband data.
- Classes.py: Implementation of different modules for action detection. 
- main.ipy : This is the main file from which you can run the experiment by specifying:
    - The user_id, handedness, position (sitting or standing).
    - The thresholds for the action detection (e.g. maximal distance between eye and hand, velocity thresholds...)
    - The data collection time in (min). Please note that the software overloads the cache memory (wth openpifpaf annotation objects) and results in a significant lag after 6min. Therefore, the experiments were run with 5 periods of 4min.

# Data preparation and classification
- lstm_module.py: contains the Dataset and Dataloader classes, a PadSequence implementation that allows to take sequences with different input lengths as well as the LSTM module implementation. Requires Pytorch-Lightning library.
- project_utils.py: contains util functions to make the train-test notebook compatible with different datasets.
- data_preparation.ipynb: loads, cleans, merges and saves files (computer vision records, ax3, mms+, applewatch), with the following parts:
    - Read seperate data files for each user.
    - Features visualization and basic statistics.
    - Visualization of the calibration signals.
    - Data labeling and saving.
 
- Sanity_checks.ipynb:  displays visualization to validate the computer vision software.
- LSTM_train_test.ipynb: loads, trains and tests the LSTM implementation. One has to specify the dataset type at the begining of the notebook ('applewatch', 'ax3', 'mms+') and the rest is done automatically.

# Main results
An LSTM implementation yields a 72.3% classification accuracy with 71.6% F1-score.

- Training and validation losses: 

![Training and validation losses](https://github.com/Omarraita/Hand-Face-Interactions-Detection/blob/main/Images/applewatch_losses.png)

- Performance Metrics: 

![Performance Metrics](https://github.com/Omarraita/Hand-Face-Interactions-Detection/blob/main/Images/applewatch_performance_metrics.png)

- Confusion Matrix:

![Confusion Matrix](https://github.com/Omarraita/Hand-Face-Interactions-Detection/blob/main/Images/applewatch_cm.png)

- Confusion Matrix normalized:

![Confusion Matrix normalized](https://github.com/Omarraita/Hand-Face-Interactions-Detection/blob/main/Images/applewatch_cm_normalized.png)

