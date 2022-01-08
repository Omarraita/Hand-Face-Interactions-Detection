# Hand-Face-Interactions-Detection

The goal of this project is to use Machine learning and Deep learning techniques to classify and detect Human Hand-Face interactions in order to warn users in case of risky hand-face interactions and especiallythe contact with mucosal membranes(e.g., Face skratching, eye-rubbing) to ultimately limit transmission of infectious diseases.  A potential application in ophthalmology: blindness is caused by diseasessuch as Keratoconus that could be prevented by eye-rubbing cessation. 

# Report
The report of this project can be found [here](https://github.com/Omarraita/MI_prediction/blob/main/MI_prediction.pdf).

# Requirements

In order to run the notebooks, it is necessary to install the following packages:

pip3 install openpifpaf==0.12.14 (For computer vision software, v0.12.14 is necessary)

pip3 install torch==1.10.0 (Pytorch Version used for training: compatible with pytorch-lightning)

pip3 install pytorch-lightning  (Practical Pytorch Library that provides built-in modules for training and performance logging)

pip3 install torch==1.10.0 (Pytorch Version used for training: compatible with pytorch-lightning)

# Data preparation

- Clone this repository into your base_folder. To run the data augmentation, download and extract Raw_Data containing raw and labelled angiography images from https://drive.switch.ch/index.php/s/367fkbeytfy24d8/authenticate (requires a password). 
- 
We expect the directory structure to be the following:

base_folder/Data/  
    codes/  
    Results/ 
  
 - The augmented data that was used to generate the results can be found in https://drive.switch.ch/index.php/s/Rh9UrhnUmVLjsFn (requires a password). 

# Models 

Link to selected Models https://drive.switch.ch/index.php/s/VsSCiQ5uY3Leikr

Link to the model that was trained with a different task (stenosis prediction), on artificial data https://drive.switch.ch/index.php/s/GLZaagFUdPobYSv

# Notebooks

- Training.ipynb : This is the main file from which you can run the main experiment:

  - Data augmentation
  
  - Train different models: {0:'model_scratch', 1:'model_pretrained', 2:'model_frangi', 3:'model_frangi_reversed', 4:'model_simsiam', 5:'model_simsiam_frangi', 6:'model_art_data'}

  - Load and test the trained models.

  - Compare the predictions between the different models.

  - Display the Gradcam visualizations.

  - Display features with TSNE.

- Verifications.ipynb : 

  - Displays the patches extracted from the original images to perform sanity checks. 

  - Verifies and data coming from the weighted dataloader

  - Contains the display_patient_views(): Visualization function that displays the different patches coming from the a given patient with the corresponding label and prediction.

- Frangi-Net.ipynb: Contains a Frangi-Net implementation attempt. Might be useful for a future work.

# Python files
- label_detection_patch_creation_.py: Runnable file that detects the labels and crops the patches from the Raw Data.
- cross_validation_lr_wd_frangi.py: Runnable file to run a cross validation on both parameters learning rate and weight decay.
- implementations.py: contains all the implemented functions for this project. The main parts are: Data preparation, Dataset Classes, Models Initializer, Train and test functions, Cross validation functions.
- utils.py:  contains util functions such as visualization functions.

# SimSiam
- Contains the adapted PyTorch implementation of SimSiam (https://github.com/leftthomas/SimSiam) to the cardio dataset. Uses some functions taken from SimCLR implementation (https://github.com/sthalles/SimCLR). 

# Results Folders
- Results: Validation results for all models.
- CV_Results: Cross validation results for the regular resnet18 model.
- CV_Results_frangi: Cross validation results for the frangi model.
- Gradcam_images: Heatmap visualizations for different models.
- test_results: Test results for all models.


# Main results
- Training and validation losses: 

![Training and validation losses](https://github.com/Omarraita/Hand-Face-Interactions-Detection/blob/main/Images/applewatch_losses.png)

- Performance Metrics: 

![Performance Metrics](https://github.com/Omarraita/Hand-Face-Interactions-Detection/blob/main/Images/applewatch_performance_metrics.png)

- Confusion Matrix:

![Confusion Matrix](https://github.com/Omarraita/Hand-Face-Interactions-Detection/blob/main/Images/applewatch_cm.png)

- Confusion Matrix normalized:

![Confusion Matrix normalized](https://github.com/Omarraita/Hand-Face-Interactions-Detection/blob/main/Images/applewatch_cm_normalized.png)

