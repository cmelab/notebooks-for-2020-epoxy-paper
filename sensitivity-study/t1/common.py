## import os

import math

import gsd
import gsd.fl
import gsd.hoomd
from scipy.signal import argrelextrema as argex
from cme_utils.analyze import autocorr
import numpy as np
import os

def get_all_maximas(lx,q,intensities):
    half_box_length = lx*0.6
    q_half_length = 2*math.pi/(half_box_length)
    peaks_q = []
    peaks_I = []
 #   print(q)
    maxima_i = argex(intensities,np.greater)[0]
    for i in maxima_i:
        if q[i] > q_half_length:
  #          print(i)
            peaks_q.append(q[i])
            peaks_I.append(intensities[i])
    #if len(peaks_I)==0:
    #    print(job)
    #    print(q_half_length,maxima_i)
    return peaks_q,peaks_I

def get_highest_maxima(lx,q,intensities):
    peaks_q,peaks_I = get_all_maximas(lx,q,intensities)
    if len(peaks_I) > 0:
        largest_peak_I = np.max(peaks_I)
        index_largest_I = peaks_I.index(largest_peak_I)
        largest_peak_q = peaks_q[index_largest_I]
    else:
        largest_peak_q=None
        largest_peak_I=None
    return largest_peak_q,largest_peak_I


def get_nth_maxima(job,q,intensities,n=1):
    '''
    Use 'n'=1 for first maxima and 'n'=2 for second maxima etc..
    '''
    peaks_q,peaks_I = get_all_maximas(job,q,intensities)
    sorted_peaks_I = np.sort(peaks_I)
    #print(peaks_q,peaks_I,sorted_peaks_I)
    nth_largest_peak_I = sorted_peaks_I[n*-1]
    index_nth_largest_I = peaks_I.index(nth_largest_peak_I)
    nth_largest_peak_q = peaks_q[index_nth_largest_I]
    return nth_largest_peak_q,nth_largest_peak_I

def savefig(plt,nbname,figname,transparent=True):
    import os
    if not os.path.exists(nbname):
        os.makedirs(nbname)
    plt.savefig(os.path.join(nbname,figname),bbox_inches='tight',transparent=transparent)
    
from mpl_toolkits.mplot3d import axes3d

class MyAxes3D(axes3d.Axes3D):

    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # set visibility of some features False 
        self.set_some_features_visibility(False)
        # draw the axes
        super(MyAxes3D, self).draw(renderer)
        # set visibility of some features True. 
        # This could be adapted to set your features to desired visibility, 
        # e.g. storing the previous values and restoring the values
        self.set_some_features_visibility(True)

        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        # disable draw grid
        zaxis.axes._draw_grid = False

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw :
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw :
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2], 
                             tmp_planes[1], tmp_planes[0], 
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes

        # disable draw grid
        zaxis.axes._draw_grid = draw_grid_old
        
import gsd
import gsd.fl
import gsd.hoomd
from scipy.signal import argrelextrema as argex
from cme_utils.analyze import autocorr
import numpy as np
import os
import math

def get_all_maximas(box_length,q,intensities):
    half_box_length = box_length*0.6
    q_half_length = 2*math.pi/(half_box_length)
    peaks_q = []
    peaks_I = []
 #   print(q)
    maxima_i = argex(intensities,np.greater)[0]
    for i in maxima_i:
        if q[i] > q_half_length:
  #          print(i)
            peaks_q.append(q[i])
            peaks_I.append(intensities[i])
    #if len(peaks_I)==0:
    #    print(job)
    #    print(q_half_length,maxima_i)
    return peaks_q,peaks_I

def get_highest_maxima(box_length,q,intensities):
    peaks_q,peaks_I = get_all_maximas(box_length,q,intensities)
    if len(peaks_I) > 0:
        largest_peak_I = np.max(peaks_I)
        index_largest_I = peaks_I.index(largest_peak_I)
        largest_peak_q = peaks_q[index_largest_I]
    else:
        largest_peak_q=None
        largest_peak_I=None
    return largest_peak_q,largest_peak_I