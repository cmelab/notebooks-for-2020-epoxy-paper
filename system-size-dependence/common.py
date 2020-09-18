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

def save_frame(job,frame):
    with gsd.hoomd.open('Frame{}.gsd'.format(frame), 'wb') as t_new:
        f = gsd.fl.GSDFile(job.fn('data.gsd'), 'rb')
        t = gsd.hoomd.HOOMDTrajectory(f)
        snap = t[frame]
        t_new.append(snap)
    #hoomd.deprecated.dump.xml(group=hoomd.group.all(), filename=job.fn('Frame{}.hoomdxml'.format(frame)), position=True)

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
        
def _get_decorrelation_time(prop_values,
                             time_steps):
    t = time_steps - time_steps[0]
    dt = t[1] - t[0]
    acorr = autocorr.autocorr1D(prop_values)
    for acorr_i in range(len(acorr)):
        if acorr[acorr_i]<0:
            break
    lags = [i*dt for i in range(len(acorr))]

    decorrelation_time = int(lags[acorr_i])
    if decorrelation_time == 0:
        decorrelation_time = 1
    decorrelation_stride = int(decorrelation_time/dt)
    nsamples = (int(t[-1])-t[0])/decorrelation_time
    temps = "There are %.5e steps, (" % t[-1]
    temps = temps + "%d" % int(t[-1])
    temps = temps + " frames)\n"
    temps = temps + "You can start sampling at t=%.5e" % t[0]
    temps = temps + " (frame %d)" % int(t[0] )
    temps = temps + " for %d samples\n" % nsamples
    temps = temps + "Because the autocorrelation time is %.5e" % lags[acorr_i]
    temps = temps + " (%d frames)\n" % int(lags[acorr_i])
    #print(temps)
    return decorrelation_time, decorrelation_stride

from scipy import stats

def get_mean_sf_from_independent_frames(job,equilibrated_percent=None):
    typeId=2
    n_views=40
    grid_size=512
    diffract_dir_pattern ='diffract_type_{}_n_views_{}_grid_size_{}_frame'.format(typeId,
                                                                                 n_views,
                                                                                  grid_size)
    directories = os.listdir(job.workspace())
    directories = [d for d in os.listdir(job.workspace()) if d.startswith(diffract_dir_pattern)]
    directories.sort(key = lambda x: int(x.split('_')[-1]))
    #print(len(directories))
    #print(directories)
    num_frames = len(directories)
    log_path = job.fn('out.log')
    data = np.genfromtxt(log_path, names=True)
    time_steps = data['timestep']
    #print('time steps',time_steps)
    if equilibrated_percent==None:
        start_i, start_t = autocorr.find_equilibrated_window(time_steps, data['potential_energy'])
    else:
        start_i=int(len(time_steps)*equilibrated_percent/100)
        start_t=time_steps[start_i]
    decorrelation_time, decorrelation_stride = _get_decorrelation_time(data['potential_energy'][start_i:], time_steps[start_i:])
    #print('decorrelation_stride:',decorrelation_stride)
    #print('decorrelation_time:',decorrelation_time)
    #print('start_i:',start_i)
    #print('start_t:',start_t)        
    if num_frames > 0:
        qs_for_all_times=[]
        Is_for_all_times=[]
        times_for_all_times=[]
        qs_list = []
        times_list = []
        Is_list = []  
        Qs_list=[]

        for i,diffract_dir in enumerate(directories):
            #print("Progress {:2.1%}".format(i / num_frames), end="\r")

            #print(diffract_dir)

            if diffract_dir.startswith(diffract_dir_pattern):
                frame = int(diffract_dir.split('_')[-1])

                decorrelated_frame_stride = int(decorrelation_time/job.sp.dcd_write)
                decorrelated_frame_stride = max(decorrelated_frame_stride,1)
                time = round(frame*job.sp.dcd_write)
                #print('decorrelated frame stride is:',decorrelated_frame_stride)
                #print('Equilibriated after time:',start_t)
                #print('time:{}, {}%{}={}'.format(time,frame,decorrelated_frame_stride,frame%decorrelated_frame_stride))
                if time >= start_t:# and frame%decorrelated_frame_stride==0:# and frame <3e6/job.sp.dcd_write:#==119 or frame==123:#%100 == 0:#num_frames/30:
                    if job.isfile('{}/asq.txt'.format(diffract_dir)):
                        data=np.genfromtxt(job.fn('{}/asq.txt'.format(diffract_dir)))

                        legend = '{} $\\Delta t(\Gamma:{})$'.format(time,job.sp.gamma)
                        qs = data[:,0]
                        Is = data[:,1] 
                        qs_for_all_times.append(qs)
                        Is_for_all_times.append(Is)
                        times_for_all_times.append(time)

                        dq=qs[1]-qs[0]
                        Is_exp = np.exp(Is)
                        q_sq = qs**2
                        Q = np.sum(Is_exp*qs*dq)
                        Qs_list.append(Q)
                        #first_peak_q,first_peak_i = get_highest_maxima(job,qs,Is)
                        #if first_peak_q >0.8 and time > 2.0e5:
                        #    first_peak_q=q_half_length

                        #qs_list.append(first_peak_q)
                        #times_list.append(time)
                        #Is_list.append(first_peak_i)
                    else:
                        print(job,'did not contain diffraction data in ',diffract_dir)
                #else:
                #    print(job,'directory {} is not as expected:{}'.format(diffract_dir,diffract_dir_pattern))
    else:
        print(job,'did not contain diffraction data for time evolution')
    print('Number of independent frames for average:',len(qs_for_all_times))
    m_q = np.mean(qs_for_all_times,axis=0)
    std_q = stats.sem(qs_for_all_times,axis=0)
    m_I = np.mean(Is_for_all_times,axis=0)
    std_I = stats.sem(Is_for_all_times,axis=0)
    return m_q,std_q,m_I,std_I

def savefig(plt,nbname,figname,transparent=True):
    import os
    if not os.path.exists(nbname):
        os.makedirs(nbname)
    plt.savefig(os.path.join(nbname,figname),bbox_inches='tight',transparent=transparent)