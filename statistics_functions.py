# Plots - Stats functions

# Functions for SMARP Matching
from datetime import datetime, timedelta
import sys
import csv
import os
import random
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score
from matplotlib.colors import LogNorm
import seaborn
import numpy as np
from scipy import stats
#from SMARPMatching_functions import log_reg_normalization_v2
import scipy
from matplotlib.patches import Polygon



# Function for calculating and printing cumulative metrics
def cumulative_metrics(output, local_labels, runs):

    CumulativeConfusion = True
    CumulativeMetrics = True
    BoxPlots = True

    # Accuracy
    accuracy = (abs(output - local_labels) < 0.5).sum(axis=1)/len(output[0])

    # Heidke Skill Score
    HSS = np.zeros(runs)
    for i in range(len(HSS)):
        tn, fp, fn, tp = confusion_matrix(local_labels[i], output[i]).ravel()
        HSS[i] = 2*(tn*tp-fn*fp)/((tn+fn)*(fn+tp)+(tn+fp)*(fp+tp))

    # True Skill Score (TSS)
    TSS = np.zeros(runs)
    for i in range(len(TSS)):
        tn, fp, fn, tp = confusion_matrix(local_labels[i], output[i]).ravel()
        if (tn+fn) == 0 or (tp+fp) == 0:
            TSS[i] = 0
        else:
            TSS[i] = (tn*tp-fp*fn)/((tn+fn)*(tp+fp))

    if CumulativeMetrics:
        print('################## Averages ##################')
        # Metrics
        print('Other Metrics')
        print('Acc:  {} +- {}'.format(np.mean(accuracy),np.std(accuracy)))
        print('TSS:  {} +- {}'.format(np.mean(TSS),np.std(TSS)))
        print('HSS:  {} +- {}'.format(np.mean(HSS),np.std(HSS)))
        #print('F1:   {}, {}, {}  (Macro,Micro,Weighted)'.format(round(f1_score(local_labels_flat, output_pred_flat, average='macro'),2), round(f1_score(local_labels_flat, output_pred_flat, average='micro'),2), round(f1_score(local_labels_flat, output_pred_flat, average='weighted'),2)))
        #print('Loss: {}          (Logarithmic)'.format(round(log_loss(local_labels_flat, output_flat),5)))
        #print('AUC:  {}          (ROC)'.format(round(roc_auc_score(local_labels_flat, output_flat),2)))

    if CumulativeConfusion:
        # Flatten
        local_labels_flat = local_labels.flatten()
        output_pred_flat = output.flatten()
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(local_labels_flat, output_pred_flat).ravel()
        data = [[tp,fp],[fn,tn]]
        conf_fig = plt.figure(1, figsize=(3, 2))
        plt.title("Confusion Matrix"); labels = ['Positive','Negative']
        seaborn.set(color_codes=True); seaborn.set(font_scale=1.4)
        ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", fmt='d', vmin=0, vmax=len(output_pred_flat)/2, annot_kws={'size': 25})
        ax.set_xticklabels(labels, fontsize=15); ax.set_yticklabels(labels, fontsize=15)
        ax.set(xlabel="Predicted Label")
        ax.set(ylabel="True Label")
        conf_fig.set_figwidth(5) 
        conf_fig.set_figheight(4) 
        plt.show()

    if BoxPlots:
        random_dists = ['ACC', 'TSS', 'HSS']
        data = [accuracy, TSS, HSS]
        fig, ax1 = plt.subplots(figsize=(12, 10))
        ax1.set_facecolor('white')
        for spine in ax1.spines.values(): spine.set_color('black')
        fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        fl = dict(marker='o', markerfacecolor='r', markersize=5)
        bp = ax1.boxplot(data, notch=0, flierprops=fl, vert=1, whis=1.5)
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax1.set_ylabel(ylabel='Score')
        num_boxes = len(data)
        # Set the axes ranges and axes labels
        ax1.set_xlim(0.5, num_boxes + 0.5)
        ax1.set_ylim(-0.2, 1.1)
        ax1.set_xticklabels(np.repeat(random_dists, 1), fontsize=16)
        plt.show()

        # Extras 
        #log_losses = np.zeros(runs); F1 = np.zeros(runs); AUC = np.zeros(runs)
        # Log Losses
        #for i in range(len(log_losses)): log_losses[i] = round(log_loss(local_labels[i], output[i]),5)
        #print('Val Losses'); print(log_losses)
        #axs[0,1].boxplot(log_losses,0,'rs',showmeans=True); axs[0,1].set_title('Validation Loss (std={})'.format(round(np.std(log_losses),2)))
        # F1 Weighted
        #for i in range(len(F1)): F1[i] = round(f1_score(local_labels[i], output[i], average='weighted'),5)
        #print('F1 Weighted'); print(F1)
        #axs[0,2].boxplot(F1,0,'rs',showmeans=True); axs[0,2].set_title('F1 Weighted (std={})'.format(round(np.std(F1),2)))
        # F1 Weighted 
        #for i in range(len(AUC)): AUC[i] = round(roc_auc_score(local_labels[i], output[i], average='weighted'),5)
        #axs[1,0].boxplot(AUC,0,'rs',showmeans=True); axs[1,0].set_title('AUC (std={})'.format(round(np.std(AUC),2)))
        
        return accuracy, HSS, TSS 



# Scatterplot for data within run (training and val distinctions too)
def svm_data_scatter(positive_train,positive_val,negative_train,negative_val,active_train,active_val,active_peak_train,active_peak_val):
    
    print('Positive dataset: {} training and {} validation datapoints'.format(len(positive_train), len(positive_val)))
    print('Negative dataset: {} training and {} validation datapoints'.format(len(negative_train), len(negative_val)))
    print('Active dataset: {} training and {} validation datapoints'.format(len(positive_train), len(positive_val)))
    print('Active Peak dataset: {} training and {} validation datapoints'.format(len(negative_train), len(negative_val)))
    for pt in positive_train: plt.scatter(pt[0],pt[1], color = 'g') # Positive Train # x:USFLUX y:MEANGBZ 
    for pv in positive_val: plt.scatter(pv[0],pv[1], color = 'g') # Positive Val # x:USFLUX y:MEANGBZ
    for nt in negative_train: plt.scatter(nt[0],nt[1], color = 'r') # Negative Train # x:USFLUX y:MEANGBZ
    for nv in negative_val: plt.scatter(nv[0],nv[1], color = 'r') # Negative Val # x:USFLUX y:MEANGBZ
    #for at in active_train: plt.scatter(at[0],at[1], color = 'm') # Negative Train # x:USFLUX y:MEANGBZ
    #for av in active_val: plt.scatter(av[0],av[1], color = 'm') # Negative Val # x:USFLUX y:MEANGBZ
    for apt in active_peak_train: plt.scatter(apt[0],apt[1], color = 'y') # Negative Train # x:USFLUX y:MEANGBZ
    for apv in active_peak_val: plt.scatter(apv[0],apv[1], color = 'y') # Negative Val # x:USFLUX y:MEANGBZ
    plt.title('Data for current SVM run'); plt.xlabel('USFLUX'); plt.ylabel('MEANGBZ')
    plt.show()



# Histogram of pure data
def data_histograms(positive, quiet, active, active_peak):

    # Create histograms
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist([positive[:,0], active_peak[:,0], quiet[:,0]], 5, label=['Positive', 'Active (Peak)', 'Quiet'])
    axs[0].legend(loc='upper right')
    axs[0].title.set_text('USFLUX')
    axs[1].hist([positive[:,1], active_peak[:,1], quiet[:,1]], 5, label=['Positive', 'Active (Peak)', 'Quiet'])
    axs[1].title.set_text('MEANGBZ')
    #axs[0].title('MEANGBZ for SEP prediction with SMARPs')
    plt.show()



def all_data_histograms(positive, quiet, active, active_peak):

    # Normalize
    all_data = positive + quiet + active + active_peak #Put together dataset
    norm_all_data = sklearn.preprocessing.normalize(all_data, axis=0) #normalize

    # Split dataset again
    positive = norm_all_data[0:len(positive)]
    quiet = norm_all_data[len(positive):(len(positive)+len(quiet))]
    active = norm_all_data[(len(positive)+len(quiet)):(len(positive)+len(quiet)+len(active))]
    active_peak = norm_all_data[(len(positive)+len(quiet)+len(active)):]

    normalized = True
    if normalized == False:
        # Create histograms for USFLUX
        fig, axs = plt.subplots(2, 4)
        fig.suptitle('USFLUX (top) and MEANGBZ (bottom)', fontsize=10)
        axs[0,0].hist(positive[:,0], 5, label=['Positive'])
        axs[0,0].title.set_text('{} Positive'.format(len(positive)))
        axs[0,1].hist(quiet[:,0], 5, label=['Quiet'])
        axs[0,1].title.set_text('{} Quiet'.format(len(quiet)))
        axs[0,2].hist(active[:,0], 5, label=['Active'])
        axs[0,2].title.set_text('{} Active'.format(len(active)))
        axs[0,3].hist(active_peak[:,0], 5, label=['Active (peak)'])
        axs[0,3].title.set_text('{} Peak Active'.format(len(active_peak)))
        # Create histograms for MEANGBZ
        axs[1,0].hist(positive[:,1], 5, label=['Positive'])
        axs[1,0].title.set_text('{} Positive'.format(len(positive)))
        axs[1,1].hist(quiet[:,1], 5, label=['Quiet'])
        axs[1,1].title.set_text('{} Quiet'.format(len(quiet)))
        axs[1,2].hist(active[:,1], 5, label=['Active'])
        axs[1,2].title.set_text('{} Active'.format(len(active)))
        axs[1,3].hist(active_peak[:,1], 5, label=['Active (peak)'])
        axs[1,3].title.set_text('{} Peak Active'.format(len(active_peak)))
    else: 
        # Normalized histograms
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].hist([positive[:,0], quiet[:,0], active_peak[:,0]], 10, label=['Positive', 'Quiet', 'Active (Peak)'],density=True)
        axs[0].legend(loc='upper right')
        axs[0].title.set_text('USFLUX')
        axs[1].hist([positive[:,1], quiet[:,1], active_peak[:,1]], 10, label=['Positive', 'Quiet', 'Active (Peak)'],density=True)
        axs[1].title.set_text('MEANGBZ')

    plt.show()



# Scatterplot for data within run WITH ANNOTATIONS
def svm_data_scatter_v2(positive_train,positive_val,active_peak_train,active_peak_val):
    
    print('Positive dataset: {} training and {} validation datapoints'.format(len(positive_train), len(positive_val)))
    print('Active dataset: {} training and {} validation datapoints'.format(len(positive_train), len(positive_val)))
    for pt in positive_train: 
        plt.scatter(pt[0],pt[4], color = 'g') # Positive Train # x:USFLUX y:MEANGBZ 
        #plt.annotate(str(round(pt[2],2)),(pt[0],pt[1]))
    for pv in positive_val: 
        plt.scatter(pv[0],pv[4], color = 'g') # Positive Val # x:USFLUX y:MEANGBZ
        #plt.annotate(str(round(pv[2],2)),(pv[0],pv[1]))
    for apt in active_peak_train: 
        plt.scatter(apt[0],apt[4], color = 'y') # Negative Train # x:USFLUX y:MEANGBZ
        #plt.annotate(str(round(apt[2],2)),(apt[0],apt[1]))
    for apv in active_peak_val: 
        plt.scatter(apv[0],apv[4], color = 'y') # Negative Val # x:USFLUX y:MEANGBZ
        #plt.annotate(str(round(apv[2],2)),(apv[0],apv[1]))
    plt.title('Data for current SVM run'); plt.xlabel('USFLUX'); plt.ylabel('Intensity')
    plt.show()



# Fitting a curve to the 1-D histograms
def best_fit(bins_,data):
    mu, sigma = scipy.stats.norm.fit(data)
    best_fit_line = scipy.stats.norm.pdf(bins_, mu, sigma)
    return best_fit_line



# 1-D Histograms 
def d1_histograms_v3(positive, active, quiet):

    bins = 20 #number of bins

    old = False
    if old == True:
        # Histograms for flux, gradient and intensity
        fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(12,4))
        _, bins_, _ = axs[0].hist([positive[:,0], active[:,0]], bins, label=['Positive', 'Active'], density = True, color = ['lightgreen','lightcoral'])
        axs[0].plot(bins_, best_fit(bins_,positive[:,0]), color = 'green'); axs[0].plot(bins_, best_fit(bins_,active[:,0]), color = 'red')
        axs[0].legend(loc='upper right'); axs[0].title.set_text('Flux')
        _, bins_, _ = axs[1].hist([positive[:,1], active[:,1]], bins, label=['Positive', 'Active'], density = True, color = ['lightgreen','lightcoral'])
        axs[1].plot(bins_, best_fit(bins_,positive[:,1]), color = 'green'); axs[1].plot(bins_, best_fit(bins_,active[:,1]), color = 'red')
        axs[1].legend(loc='upper right'); axs[1].title.set_text('Vertical Field Gradient')
        _, bins_, _ = axs[2].hist([positive[:,4], active[:,4]], bins, label=['Positive', 'Active'], density = True, color = ['lightgreen','lightcoral'])
        axs[2].plot(bins_, best_fit(bins_,positive[:,4]), color = 'green'); axs[2].plot(bins_, best_fit(bins_,active[:,4]), color = 'red')
        axs[2].legend(loc='upper right'); axs[2].title.set_text('Flare Intensity')
        plt.show()

        # Histograms for the two angular distances and 
        fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(8,4))
        _, bins_, _ = axs[0].hist([positive[:,2], active[:,2]], bins, label=['Positive', 'Active'], density = True, color = ['lightgreen','lightcoral'])
        axs[0].plot(bins_, best_fit(bins_,positive[:,2]), color = 'green'); axs[0].plot(bins_, best_fit(bins_,active[:,2]), color = 'red')
        axs[0].legend(loc='upper right'); axs[0].title.set_text('AR Angular Distance')
        _, bins_, _ = axs[1].hist([positive[:,3], active[:,3]], bins, label=['Positive', 'Active'], density = True, color = ['lightgreen','lightcoral'])
        axs[1].plot(bins_, best_fit(bins_,positive[:,3]), color = 'green'); axs[1].plot(bins_, best_fit(bins_,active[:,3]), color = 'red')
        axs[1].title.set_text('Flare Angular Distance'); axs[1].legend(loc='upper right')
        plt.show()

    translucent = True
    if translucent == True:
        bins_num = 25 #number of bins
        opacity_pos = 1; opacity_neg = 0.7; color_pos = 'teal'; color_neg = '#DC143C' ##DC143C crimson
        #'#f2c649','blue' #'steelblue'; color_neg = 'darkorange'
        # Histograms for flux, gradient and intensity
        #matplotlib.rcParams['text.usetex'] = True
        from matplotlib.font_manager import findfont, FontProperties
        font = findfont(FontProperties(family=['sans-serif']))
        
        fig, axs = plt.subplots(2, 4, tight_layout=True, figsize=(15,6))
        bins=np.histogram(np.hstack((positive[:,0],active[:,0])), bins=30)[1]
        axs[0,0].hist(positive[:,0], bins, label = 'Positive', density = True, color = color_pos, alpha = opacity_pos)
        axs[0,0].hist(active[:,0], bins, label = 'Negative', density = True, color = color_neg, alpha = opacity_neg)
        #axs[0,0].legend(loc='upper right'); 
        axs[0,0].title.set_text('Total Unsigned Flux'); axs[0,0].set(ylabel='Data Density',xlabel=r'$log(Mx)$')
        axs[0,0].set_xlim([21.2, 23.4])
        
        bins=np.histogram(np.hstack((positive[:,1],active[:,1])), bins=bins_num)[1]
        axs[0,1].hist(positive[:,1], bins, label = 'Positive', density = True, color = color_pos, alpha = opacity_pos)
        axs[0,1].hist(active[:,1], bins, label= 'Negative', density = True, color = color_neg, alpha = opacity_neg)
        #axs[0,1].legend(loc='upper right'); 
        axs[0,1].title.set_text('Vertical Field Gradient'); axs[0,1].set(xlabel=r'$G/mm$')

        bins=np.histogram(np.hstack((positive[:,6],active[:,6])), bins=30)[1]
        axs[0,2].hist(positive[:,6], bins, label = 'Positive', density = True, color = color_pos, alpha = opacity_pos)
        axs[0,2].hist(active[:,6], bins, label= 'Negative', density = True, color = color_neg, alpha = opacity_neg)
        #axs[0,2].legend(loc='upper right'); 
        axs[0,2].title.set_text('R Value'); axs[0,2].set(xlabel=r'$log(Mx)$')
        axs[0,2].set_xlim([1.9, 5.6])

        bins=np.histogram(np.hstack((positive[:,2],active[:,2])), bins=bins_num)[1]
        axs[1,0].hist(positive[:,2], bins, label = 'Positive', density = True, color = color_pos, alpha = opacity_pos)
        axs[1,0].hist(active[:,2], bins, label = 'Negative', density = True, color = color_neg, alpha = opacity_neg)
        #axs[1,0].legend(loc='upper right'); 
        axs[1,0].title.set_text('Active Region Angular Distance'); axs[1,0].set(ylabel='Data Density',xlabel=r'$rad$')
        axs[1,0].set_xlim([-2.7, 1])
        
        bins=np.histogram(np.hstack((positive[:,5],active[:,5])), bins=40)[1]
        axs[1,1].hist(positive[:,5], bins, label = 'Positive', density = True, color = color_pos, alpha = opacity_pos)
        axs[1,1].hist(active[:,5], bins, label = 'Negative', density = True, color = color_neg, alpha = opacity_neg)
        axs[1,1].legend(loc='upper right'); 
        axs[1,1].title.set_text('Active Region Area'); axs[1,1].set(xlabel=r'$deg^2$')
        axs[1,1].set_xlim([0, 3000])

        bins=np.histogram(np.hstack((positive[:,3],active[:,3])), bins=bins_num)[1]
        axs[1,2].hist(positive[:,3], bins, label = 'Positive', density = True, color = color_pos, alpha = opacity_pos)
        axs[1,2].hist(active[:,3], bins, label = 'Negative', density = True, color = color_neg, alpha = opacity_neg)
        #axs[1,2].legend(loc='upper right'); 
        axs[1,2].title.set_text('Flare Angular Distance'); axs[1,2].set(xlabel=r'$rad$')

        bins=np.histogram(np.hstack((positive[:,4],active[:,4])), bins=bins_num)[1]
        axs[1,3].hist(positive[:,4], bins, label = 'Positive', density = True, color = color_pos, alpha = opacity_pos)
        axs[1,3].hist(active[:,4], bins, label= 'Negative', density = True, color = color_neg, alpha = opacity_neg)
        #axs[0,2].legend(loc='upper right'); 
        axs[1,3].title.set_text('Flare Intensity'); axs[1,3].set(xlabel=r'$log(W/m^2)$')

        plt.show()



# Ranges for 2-D histograms
def hist_range(data1x,data1y,data2x,data2y):

    xmin = min(float(np.min(data1x)),float(np.min(data2x)))
    xmax = max(float(np.max(data1x)),float(np.max(data2x)))
    ymin = min(float(np.min(data1y)),float(np.min(data2y)))
    ymax = max(float(np.max(data1y)),float(np.max(data2y)))

    return [[xmin, xmax], [ymin, ymax]]



# 2-D Histograms
def d2_histograms_v3(positive, active, quiet):
    
    bins = 200 #number of bins
    
    # Flux vs. Gbz
    fig,axs = plt.subplots(1, 2, tight_layout=True, figsize=(8,4))
    data_range = hist_range(positive[:,0], positive[:,1], active[:,0], active[:,1])
    axs[0].hist2d(positive[:,0], positive[:,1], bins=bins, density=True, range = data_range)
    axs[0].title.set_text('Positive'); axs[0].set_xlabel('Flux'); axs[0].set_ylabel('Vertical Field Gradient')
    axs[1].hist2d(active[:,0], active[:,1], bins=bins, density=True, range = data_range)
    axs[1].title.set_text('Active'); axs[1].set_xlabel('Flux'); axs[1].set_ylabel('Vertical Field Gradient')
    plt.show()

    # Flux vs. Angular Dist (AR)
    fig,axs = plt.subplots(1, 2, tight_layout=True, figsize=(8,4))
    data_range = hist_range(positive[:,0], positive[:,2], active[:,0], active[:,2])
    axs[0].hist2d(positive[:,0], positive[:,2], bins=bins, density=True, range = data_range)
    axs[0].title.set_text('Positive'); axs[0].set_xlabel('Flux'); axs[0].set_ylabel('Angular Distance (AR)')
    axs[1].hist2d(active[:,0], active[:,2], bins=bins, density=True, range = data_range)
    axs[1].title.set_text('Active'); axs[1].set_xlabel('Flux'); axs[1].set_ylabel('Angular Distance (AR)')
    plt.show()

    # Flux vs. Angular Dist (flare)
    fig,axs = plt.subplots(1, 2, tight_layout=True, figsize=(8,4))
    data_range = hist_range(positive[:,0], positive[:,3], active[:,0], active[:,3])
    axs[0].hist2d(positive[:,0], positive[:,3], bins=bins, density=True, range = data_range)
    axs[0].title.set_text('Positive'); axs[0].set_xlabel('Flux'); axs[0].set_ylabel('Angular Distance (flare)')
    axs[1].hist2d(active[:,0], active[:,3], bins=bins, density=True, range = data_range)
    axs[1].title.set_text('Active'); axs[1].set_xlabel('Flux'); axs[1].set_ylabel('Angular Distance (flare)')
    plt.show()

    # Flux vs. Flare Intensity
    fig,axs = plt.subplots(1, 2, tight_layout=True, figsize=(8,4))
    data_range = hist_range(positive[:,0], positive[:,4], active[:,0], active[:,4])
    h = axs[0].hist2d(positive[:,0], positive[:,4], bins=50, density=True, range = data_range, norm=LogNorm())
    axs[0].title.set_text('Positive'); axs[0].set_xlabel('Flux'); axs[0].set_ylabel('Flare Intensity')
    h = axs[1].hist2d(active[:,0], active[:,4], bins=bins, density=True, range = data_range, norm=LogNorm())
    axs[1].title.set_text('Active'); axs[1].set_xlabel('Flux'); axs[1].set_ylabel('Flare Intensity')
    plt.colorbar(h[3])
    plt.show() 


    # Flux vs. Flare Intensity SCATTER
    fig,axs = plt.subplots(1, 2, tight_layout=True, figsize=(8,4))
    axs[0].scatter(np.log(positive[:,0]), np.log(positive[:,4]))
    axs[0].title.set_text('Positive'); axs[0].set_xlabel('Flux'); axs[0].set_ylabel('Flare Intensity')
    axs[0].set_ylim([-5, -18]); axs[0].set_xlim([42, 54]); axs[0].set_xscale('log')
    axs[1].scatter(np.log(active[:,0]), np.log(active[:,4]))
    axs[1].title.set_text('Active'); axs[1].set_xlabel('Flux'); axs[1].set_ylabel('Flare Intensity')
    axs[1].set_ylim([-5, -18]); axs[1].set_xlim([42, 54]); axs[1].set_xscale('log')
    plt.show()

    # Angular Dist (AR) vs. Angular Dist (flare)
    fig,axs = plt.subplots(1, 2, tight_layout=True, figsize=(8,4))
    data_range = hist_range(positive[:,2], positive[:,3], active[:,2], active[:,3])
    axs[0].hist2d(positive[:,2], positive[:,3], bins=bins, density=True, range = data_range)
    axs[0].title.set_text('Positive'); axs[0].set_xlabel('Angular Distance (AR)'); axs[0].set_ylabel('Angular Distance (flare)')
    axs[1].hist2d(active[:,2], active[:,3], bins=bins, density=True, range = data_range)
    axs[1].title.set_text('Active'); axs[1].set_xlabel('Angular Distance (AR)'); axs[1].set_ylabel('Angular Distance (flare)')
    plt.show()

    # Angular Dist (AR) vs. Flare Intensity
    fig,axs = plt.subplots(1, 2, tight_layout=True, figsize=(8,4))
    data_range = hist_range(positive[:,2], positive[:,4], active[:,2], active[:,4])
    axs[0].hist2d(positive[:,2], positive[:,4], bins=bins, density=True, range = data_range)
    axs[0].title.set_text('Positive'); axs[0].set_xlabel('Angular Dist (AR)'); axs[0].set_ylabel('Flare Intensity')
    axs[1].hist2d(active[:,2], active[:,4], bins=bins, density=True, range = data_range)
    axs[1].title.set_text('Active'); axs[1].set_xlabel('Angular Dist (AR)'); axs[1].set_ylabel('Flare Intensity')
    plt.show() 

    # Angular Dist (flare) vs. Flare Intensity
    fig,axs = plt.subplots(1, 2, tight_layout=True, figsize=(8,4))
    data_range = hist_range(positive[:,3], positive[:,4], active[:,3], active[:,4])
    axs[0].hist2d(positive[:,3], positive[:,4], bins=bins, density=True, range = data_range)
    axs[0].title.set_text('Positive'); axs[0].set_xlabel('Angular Dist (flare)'); axs[0].set_ylabel('Flare Intensity')
    axs[1].hist2d(active[:,3], active[:,4], bins=bins, density=True, range = data_range)
    axs[1].title.set_text('Active'); axs[1].set_xlabel('Angular Dist (flare)'); axs[1].set_ylabel('Flare Intensity')
    plt.show() 



def t_test(positive, active, quiet):

    Predictor = ['Flux:    ','VF Grad:  ','AR Dist:  ','Fl Dist:  ','FL Int:   ','AR Area:  ','R Value:  ']
    for i in range(len(Predictor)):
        T = stats.ttest_ind(positive[:,i],active[:,i])
        print('{}Statistic = {} / P = {}'.format(Predictor[i],round(abs(T[0]),2),abs(T[1])))



def graph_SMARP_sequences_positive(flux,lon,time,data,sep):

    if (len(flux) != len(lon)) and (len(lon) != len(time)): print('ERROR') # Error message if there is size missmatch
   
    all_together = False
    if all_together == True:
        for i in range(len(flux)):
            flux_seq = [float('nan') if x==float(0) else np.log(x) for x in flux[i]]
            lon_seq = [float('nan') if x==float(0) else x for x in lon[i]]
            plt.plot(lon_seq,flux_seq)
            plt.scatter([float('nan'),data[i][5],sep[i][0]],[float('nan'),np.log(float(data[i][0])),np.log(float(sep[i][1]))]) # SMARP data point
        plt.show()

    every_year = True
    if every_year == True:
        previous_year = time[0].year
        plt.figure(figsize=(12,6))
        for i in range(len(flux)):
            if time[i].year != previous_year:
                plt.title('Positive '+str(time[i-1].year)); plt.xlabel('Longitude'); plt.ylabel('log(Flux)'); plt.show()
                previous_year = time[i].year
                plt.figure(figsize=(12,6)) 
            flux_seq = [float('nan') if x==float(0) else np.log(x) for x in flux[i]]
            lon_seq = [float('nan') if x==float(0) else x for x in lon[i]]
            plt.plot(lon_seq,flux_seq)
            plt.scatter([float('nan'),data[i][5],sep[i][0]],[float('nan'),np.log(float(data[i][0])),np.log(float(sep[i][1]))]) # SMARP data point
            


def graph_SMARP_sequences_negative(flux,lon,time,flare):

    
    if (len(flux) != len(lon)) and (len(lon) != len(time)): print('ERROR') # Error message if there is size missmatch

    previous_year = time[0].month
    plt.figure(figsize=(12,6))
    for i in range(len(flux)):
        if time[i].month != previous_year:
            plt.title('Negative '+str(time[i-1].month)+' - '+str(time[i-1].year)); plt.xlabel('Longitude'); plt.ylabel('log(Flux)'); plt.show()
            previous_year = time[i].month
            plt.figure(figsize=(12,6)) 
        flux_seq = [float('nan') if float(x)==float(0) else np.log(float(x)) for x in flux[i]]
        lon_seq = [float('nan') if float(x)==float(0) else float(x) for x in lon[i]]
        plt.plot(lon_seq,flux_seq)
        plt.scatter([float('nan'),flare[i][0]],[float('nan'),np.log(float(flare[i][1]))]) # SMARP data point