import pickle
from pylab import gca
import numpy as np
import matplotlib.pyplot as plt 
import os 
import math 

def plot_performance(base_folder, data_name, num_epochs, hist_test, hist_train, f1_score, loss_train, loss_val, save=False):
    """
    Plots and saves the model's performances: train and validation losses, train and validation accuracies, specificity, sensitivity and f1_score.
    """
    if not os.path.exists(base_folder+'Results'):
      os.makedirs(base_folder+'Results')
    
    fig = plt.figure()
    plt.plot(loss_val, color = 'r', alpha= 0.7, label = 'Validation loss')
    plt.plot(loss_train, color = 'b', alpha= 0.7, label='Training loss')
    plt.xlabel('Epochs',fontweight="bold", fontfamily='serif')
    plt.ylabel('Cross Entropy Loss',fontweight="bold", fontfamily='serif')
    plt.grid()
    plt.xticks(np.arange(1, num_epochs+1, math.floor(num_epochs/10)))
    ax = gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        tick.label1.set_fontsize(9)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        tick.label1.set_fontsize(9)

    plt.legend(prop={'family':'serif','weight':'bold'})
    if(save):
        plt.savefig(base_folder+'Results/'+data_name+'_losses.png')
    
    fig = plt.figure()
    plt.plot(hist_test,color = 'r', alpha= 0.7, label = 'Validation accuracy')
    plt.plot(hist_train, color = 'b', alpha= 0.7, label = 'Training accuracy')
    plt.plot(f1_score, color = 'black', alpha= 0.8, label = 'F1-score')
    plt.xlabel('Epochs',fontweight="bold", fontfamily='serif')
    plt.grid()
    plt.xticks(np.arange(1, num_epochs+1, math.floor(num_epochs/10)))
    ax = gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        tick.label1.set_fontsize(9)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        tick.label1.set_fontsize(9)

    plt.legend(prop={'family':'serif','weight':'bold'})
    if(save):
        plt.savefig(base_folder+'Results/'+data_name+'_performance_metrics.png')

def select_data(base_folder, dataset_name):
  # Apple watch
  features_applewatch = ['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
                        'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)', 'motionRotationRateX(rad/s)',
                          'motionRotationRateY(rad/s)', 'motionRotationRateZ(rad/s)', 'motionUserAccelerationX(G)',
                          'motionUserAccelerationY(G)', 'motionUserAccelerationZ(G)', 'motionQuaternionX(R)',
                        'motionQuaternionY(R)', 'motionQuaternionZ(R)', 'motionQuaternionW(R)', 'motionGravityX(G)',
                        'motionGravityY(G)', 'motionGravityZ(G)']
  # MMS+
  features_mms = ['x-axis (T)', 'y-axis (T)', 'z-axis (T)',
              'x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)',
              'x-axis (g)', 'y-axis (g)', 'z-axis (g)']
  # Ax3
  features_ax3 = ['acc_x','acc_y','acc_z']

  # Create dict 
  options = {'data': [], 'features': [], 'data_path': [],  'log_path': [], 'model_path': []}

  options['data'] = ['applewatch', 'mms+', 'ax3']

  options['features'] = [features_applewatch, features_mms, features_ax3]

  options['data_path'] = [base_folder+'Data/labeled_data_apple.csv',
                        base_folder+'Data/labeled_data_mms.csv',
                        base_folder+'Data/labeled_data_ax3.csv']

  options['log_path'] = [base_folder+'Data/applewatch_log',
                        base_folder+'Data/mms_log',
                        base_folder+'Data/ax3_log']

  options['model_path'] = [base_folder+'Data/applewatch_log/models/model.pth',
                        base_folder+'Data/mms_log/models/model.pth',
                        base_folder+'Data/ax3_log/models/model.pth']
  index = np.where(np.array(options['data']) == dataset_name)[0].item()

  return options["features"][index], options["data_path"][index], options["log_path"][index], options["model_path"][index]