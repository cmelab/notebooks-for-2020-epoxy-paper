import cme_utils
from cme_utils.analyze import autocorr
import numpy as np
import matplotlib.pyplot as plt


def get_split_quench_job_msd(job,prop_name):
    times = []
    prop_vals = []
    qTs=[]
    if job.isfile('msd.log'):
        log_path = job.fn('msd.log')
        data = np.genfromtxt(log_path, names=True)
        PROP_NAME =prop_name
        prop_values = data[PROP_NAME]#'pair_lj_energy']
        time_steps = data['timestep']
        len_prof = len(job.sp.quench_temp_prof)
        for i in range(0,len_prof,2):
            current_point = job.sp.quench_temp_prof[i]
            next_point = job.sp.quench_temp_prof[i+1]
            start_time = current_point[0]
            end_time = next_point[0]
            if current_point[1]!=next_point[1]:
                print('WARNING! Detected a non isothermal step')
            target_T = current_point[1]
            #print(time_steps)
            #print(start_time,end_time)
            indices = np.where((time_steps>=start_time)&(time_steps<=end_time))
            start_index = indices[0][0]
            end_index = indices[0][-1]
            sliced_ts = time_steps[start_index:end_index+1]
            sliced_prop_vals = prop_values[start_index:end_index+1]
            #sliced_pe = pe[start_index:end_index+1]
            #mean,std = get_mean_and_std(job,sliced_ts,sliced_prop_vals,sliced_pe)
            #means.append(mean)
            #stds.append(std)
            times.append(sliced_ts)
            prop_vals.append(sliced_prop_vals)
            qTs.append(target_T)
    return times,prop_vals,qTs

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

def get_mean_and_std_from_time_step(job, time_steps, prop_values,start_t):
    start_i = np.where(time_steps >= start_t)[0]
    if len(start_i) >0:
        start_i=start_i[0]
    else:
        start_i = 0
        
    if start_i < len(time_steps):
        independent_vals_i = np.arange(start_i, len(prop_values)-1, 1)
        independent_vals = prop_values[independent_vals_i]
        #print(independent_vals)
        mean=np.mean(independent_vals)
        std=np.std(independent_vals)
    else:
        print('the {} values given have not reached equilibrium.'.format(prop))
        mean = None
        std = None
    return mean, std

def get_mean_and_std(job, time_steps, prop_values,pe,mean_from_second_half=False):
    if mean_from_second_half:
        start_i = int(len(time_steps)*0.75)
        start_t = time_steps[start_i]
    else:
        start_i, start_t = autocorr.find_equilibrated_window(time_steps, pe)
            
    if start_i < len(time_steps):
        decorrelation_time, decorrelation_stride = _get_decorrelation_time(prop_values[start_i:], time_steps[start_i:])
        #print('decorrelation_time:',decorrelation_time)
        independent_vals_i = np.arange(start_i, len(prop_values)-1, decorrelation_stride)
        independent_vals = prop_values[independent_vals_i]
        #print(independent_vals)
        mean=np.mean(independent_vals)
        std=np.std(independent_vals)
    else:
        print('the {} values for {} given have not reached equilibrium.'.format(job,prop))
        mean = None
        std = None
    return mean, std


        
def get_mean_and_std_from_log(job, prop):
    if job.isfile('out.log'):
        log_path = job.fn('out.log')
        data = np.genfromtxt(log_path, names=True)
        prop_values = data[prop]
        time_steps = data['timestep']
        start_i, start_t = autocorr.find_equilibrated_window(time_steps, prop_values)
        if start_i < len(time_steps):
            decorrelation_time, decorrelation_stride = _get_decorrelation_time(prop_values[start_i:], time_steps[start_i:])
            independent_vals_i = np.arange(start_i, len(prop_values)-1, decorrelation_stride)
            independent_vals = prop_values[independent_vals_i]
            #print(independent_vals)
            mean=np.mean(independent_vals)
            std=np.std(independent_vals)
        else:
            print('the {} values given have not reached equilibrium.'.format(prop))
            mean = None
            std = None
    else:
        print('could not find log file for {}'.format(job))
        mean=None
        std=None
    #print(mean)
    return mean, std


def plot_equilibriation(df_filtered,
                        project,
                        prop_name,
                        draw_decorrelated_samples=False,
                        draw_equilibrium_window=True,
                        mean_from_second_half=False):
    df_sorted = df_filtered.sort_values(by=['quench_T'])
    df_grouped = df_sorted.groupby('quench_T')


    quenchTs=[]
    mean_vols=[]
    vol_stds=[]
    colors = plt.cm.plasma(np.linspace(0,1,len(df_grouped)))
    i=0
    for name,group in df_grouped:
        time_steps_temp = []
        mean_vals_temp = []
        val_stds_temp = []
        for job_id in group.index:
    #for i,job_id in enumerate(df_sorted.index):
            job = project.open_job(id=job_id)
            #print(job)
            if job.isfile('out.log'):
                log_path = job.fn('out.log')
                data = np.genfromtxt(log_path, names=True)
                PROP_NAME =prop_name
                prop_values = data[PROP_NAME]#'pair_lj_energy']
                time_steps = data['timestep']
                if mean_from_second_half:
                    start_i = int(len(time_steps)*.75)
                    #print(job,'start_i',start_i,len(time_steps),time_steps)
                    start_t = time_steps[start_i]
                else:
                    start_i, start_t = autocorr.find_equilibrated_window(time_steps, data['potential_energy'])
                decorrelation_time, decorrelation_stride = _get_decorrelation_time(data['potential_energy'][start_i:], time_steps[start_i:])
                #print('decorrelation_time:',decorrelation_time)
                independent_vals_i = np.arange(start_i, len(prop_values)-1, decorrelation_stride)
                independent_vals = time_steps[independent_vals_i]
                #starttime_steps.index(start_t)

                if 'quench_T' in job.sp:
                    label = 'q_T:{},cure:{}'.format(job.sp.quench_T,job.sp.stop_after_percent)
                    #label = 'tau:{}, tauP:{}'.format(job.sp.tau,job.sp.tauP)
                else:
                    label = 'kT:{},cure:{}'.format(job.sp.kT,job.sp.stop_after_percent)
                time_steps_temp.append(time_steps)
                mean_vals_temp.append(prop_values)
            else:
                print('did not find out.log for',job)
        mean_time_steps = np.mean(time_steps_temp,axis=0)
        mean_prop_values = np.mean(mean_vals_temp,axis=0)
        plt.plot(mean_time_steps,mean_prop_values,label=label,color=colors[i],linewidth=1.0)
        i+=1
        if draw_decorrelated_samples:
            for xval in independent_vals:
                plt.axvline(x=xval,linestyle='--',linewidth=0.2)
        if draw_equilibrium_window:
            plt.plot(mean_time_steps[start_i],
                     mean_prop_values[start_i],
                     marker='*',
                     color='r',
                     markersize=10)
            #print(time_steps)
            #decorr_i = np.where(time_steps >= decor_time)[0][0]
            #print(decorr_
            


def get_values_for_quenchTs(df_filtered,project, prop,mean_from_second_half=False):
    df_sorted = df_filtered.sort_values(by=['quench_T'])
    df_grouped = df_sorted.groupby('quench_T')
    quenchTs=[]
    mean_vals=[]
    val_stds=[]
    for name,group in df_grouped:
        quench_Ts_temp = []
        mean_vals_temp = []
        val_stds_temp = []
        
        for job_id in group.index:
            #job_id = group.signac_id
            #print(name,job_id)
            job = project.open_job(id=job_id)
            #print(job)
            if job.isfile('out.log'):
                log_path = job.fn('out.log')
                data = np.genfromtxt(log_path, names=True)
                prop_value = data[prop]
                time_steps = data['timestep']
                pe = data['potential_energy']
                #print(job)
                mean,std = get_mean_and_std(job,time_steps,prop_value,pe,mean_from_second_half)
                if mean is not None:
                    quench_Ts_temp.append(job.sp.quench_T)
                    mean_vals_temp.append(mean)
                    val_stds_temp.append(std)

        quenchTs.append(np.mean(quench_Ts_temp))
        mean_vals.append(np.mean(mean_vals_temp))
        val_stds.append(np.mean(val_stds_temp))
    return quenchTs,mean_vals,val_stds     


def line_intersect(m1, b1, m2, b2):
    if m1 == m2:
        print ("These lines are parallel!!!")
        return None
    # y = mx + b
    # Set both lines equal to find the intersection point in the x direction
    # m1 * x + b1 = m2 * x + b2
    # m1 * x - m2 * x = b2 - b1
    # x * (m1 - m2) = b2 - b1
    # x = (b2 - b1) / (m1 - m2)
    x = (b2 - b1) / (m1 - m2)
    # Now solve for y -- use either line, because they are equal here
    # y = mx + b
    y = m1 * x + b1
    return x,y

from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from piecewise.regressor import piecewise #https://www.datadoghq.com/blog/engineering/piecewise-regression/
from piecewise.plotter import plot_data_with_regression

def DiBenedetto(alphas,T1,T0,inter_param):
    Tgs = []
    for alpha in alphas:
        Tg = inter_param*alpha*(T1-T0)/(1-(alpha*(1-inter_param))) +T0
        Tgs.append(Tg)
    return Tgs

def fit_Tg_to_DiBenedetto(alphas,Tgs,T1,T0=None):
    import warnings
    np.seterr(all='raise')
    plot_fit_fails=True
    inter_parm=0.5
    try:
        if T1==None and T0==None:
            smallestTg=Tgs[0]
            largestTg=Tgs[-1]
            popt, pcov = curve_fit(lambda Xs,T1,T0: DiBenedetto(Xs,T1,T0,inter_parm),
                               alphas,Tgs,
                               #p0=[0,0],
                               p0=[largestTg,smallestTg],
                               #bounds=([-np.infty,-np.infty],[np.infty,np.infty])
                               bounds=([0,0],[largestTg*1.5,smallestTg*1.2]))#,maxfev=200000)
        elif T1==None and T0!=None:
            popt, pcov = curve_fit(lambda Xs,T1: DiBenedetto(Xs,T1,T0,inter_parm),
                               alphas,Tgs,
                               #p0=[0,0],
                               p0=[1],
                               #bounds=([-np.infty,-np.infty],[np.infty,np.infty])
                               bounds=([0],[np.infty]))#,maxfev=200000)
        else:
            popt, pcov = curve_fit(lambda Xs,T0: DiBenedetto(Xs,T1,T0,inter_parm),
                                   alphas,Tgs,
                                   #p0=[0,0],
                                   p0=[0],
                                   #bounds=([-np.infty,-np.infty],[np.infty,np.infty])
                                   bounds=([-np.infty],[np.infty]))#,maxfev=200000)
        #print('found fit')
    except FloatingPointError:
        print('Curve fitting failed(FloatingPointError)')
    except RuntimeError:
        print('Curve fitting failed(RuntimeError)')
    except TypeError:
        print('Curve fitting failed(TypeError)')
    except ValueError:
        print('Curve fitting failed(ValueError)')

    ydata = np.asarray(Tgs)
    if T1==None and T0==None:
        fit_ydata = DiBenedetto(alphas,*popt,inter_parm)
    elif T1==None and T0!=None:
        fit_ydata = DiBenedetto(alphas,*popt,T0,inter_parm)
    else:
        fit_ydata = DiBenedetto(alphas,T1,*popt,inter_parm)
    residuals = ydata - fit_ydata
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    #print('ss_res',ss_res,'ss_tot',ss_tot)
    if ss_tot == 0:
        #print('found ss_tot: 0')
        r_squared = 0
    else:
        r_squared = 1 - (ss_res / ss_tot)
    if T1==None and T0==None:
        return r_squared,fit_ydata,popt[0],inter_parm,popt[1]
    else:
        return r_squared,fit_ydata,popt[0],inter_parm#,popt[1]

def find_Tg(quenchTs, mean_vals,sap):
    print(sap)
    if True:#sap<=50.:
        use_first_deviation = False
        if use_first_deviation:
            model = piecewise(quenchTs, mean_vals)
            if len(model.segments) == 2:
                lines = []
                l1 = model.segments[0]
                m1 = l1.coeffs[1]
                b1 = l1.coeffs[0]
                l2 = model.segments[1]
                m2 = l2.coeffs[1]
                b2 = l2.coeffs[0]
            f = InterpolatedUnivariateSpline(quenchTs, mean_vals, k=2)
            dxdT = f.derivative(n=1)
            dx_dTs = dxdT(quenchTs)
            dev_index = np.where(np.abs(dx_dTs)>m1)[0][0]
            x=quenchTs[dev_index]
            y=mean_vals[dev_index]
        else:
            print('using derivatives')
            f = InterpolatedUnivariateSpline(quenchTs, mean_vals, k=2)
            dxdT = f.derivative(n=1)
            d2xdT = f.derivative(n=2)
            dx_dTs = dxdT(quenchTs)
            d2x_dT2s = d2xdT(quenchTs)
            max_dx2 = np.max(d2x_dT2s)
            min_dx2 = np.min(d2x_dT2s)
            max_i = np.where(d2x_dT2s==max_dx2)[0][0]
            min_i = np.where(d2x_dT2s==min_dx2)[0][0]
            x = (quenchTs[min_i]+quenchTs[max_i])/2
            y = (mean_vals[min_i]+mean_vals[max_i])/2
    else:
        print('using line iftting')
        #plot_data_with_regression(quenchTs, mean_vals)
        model = piecewise(quenchTs, mean_vals)
        #print(model) 
        if len(model.segments) == 2:
            lines = []
            l1 = model.segments[0]
            m1 = l1.coeffs[1]
            b1 = l1.coeffs[0]
            l2 = model.segments[1]
            m2 = l2.coeffs[1]
            b2 = l2.coeffs[0]
            x,y = line_intersect(m1,b1,m2,b2)
            
        else:
            print('WARNING: found more or less than 2 line segments in regression!')
    return x,y

def plot_this(job,time_steps,prop_values,pe,color,label=None,normalize_by_mean=False,mean_from_second_half=True):
    if mean_from_second_half:
        start_i = int(len(time_steps)*.75)
        start_t = time_steps[start_i]
    else:
        start_i, start_t = autocorr.find_equilibrated_window(time_steps, pe)
    decorrelation_time, decorrelation_stride = _get_decorrelation_time(prop_values[start_i:], time_steps[start_i:])
    #print('decorrelation_time:',decorrelation_time)
    independent_vals_i = np.arange(start_i, len(prop_values)-1, decorrelation_stride)
    independent_vals = time_steps[independent_vals_i]
    #starttime_steps.index(start_t)
    #for xval in independent_vals_i:
    #    plt.axvline(x=xval,linestyle='--',linewidth=0.2)
    indices = list(range(0,len(prop_values)))
    if len(indices) != len(prop_values):
        print('Check the length of arrays')
    #print(indices)
    #print(prop_values)
    if normalize_by_mean:
        mean,std = get_mean_and_std(job,time_steps,prop_values,pe)
        prop_values = prop_values/mean
        plt.axhline(y=1.0,linewidth=1.0,linestyle='--')
    plt.plot(indices,prop_values,label=label,linewidth=1,color=color)
    plt.plot(start_i,prop_values[start_i],marker='*',color='r', markersize=10)
                
def get_split_quench_job_property_mean_std(job,prop_name):
    means = []
    stds = []
    times = []
    temps = []
    if job.isfile('out.log'):
        log_path = job.fn('out.log')
        data = np.genfromtxt(log_path, names=True)
        PROP_NAME =prop_name
        prop_values = data[PROP_NAME]#'pair_lj_energy']
        time_steps = data['timestep']
        pe = data['potential_energy']
        print(job)
        len_prof = len(job.sp.quench_temp_prof)
        for i in range(0,len_prof,2):
            current_point = job.sp.quench_temp_prof[i]
            next_point = job.sp.quench_temp_prof[i+1]
            start_time = current_point[0]
            end_time = next_point[0]
            if current_point[1]!=next_point[1]:
                print('WARNING! Detected a non isothermal step')
            target_T = current_point[1]
            #print(time_steps)
            #print(start_time,end_time)
            indices = np.where((time_steps>=start_time)&(time_steps<=end_time))
            start_index = indices[0][0]
            end_index = indices[0][-1]
            sliced_ts = time_steps[start_index:end_index+1]
            sliced_prop_vals = prop_values[start_index:end_index+1]
            sliced_pe = pe[start_index:end_index+1]
            mean,std = get_mean_and_std(job,sliced_ts,sliced_prop_vals,sliced_pe)
            means.append(mean)
            stds.append(std)
            times.append((start_time,end_time))
            temps.append(target_T)
    return means,stds,times,temps
            
def split_log(df_filtered,project,prop_name,filter_temp,rtol=0.1,show_all=True,normalize_by_mean=False):
    df_sorted = df_filtered.sort_values(by=['quench_T'])
    
    
    for job_id in df_sorted.index:
        job = project.open_job(id=job_id)
        #print(job)
        if job.isfile('out.log'):
            log_path = job.fn('out.log')
            data = np.genfromtxt(log_path, names=True)
            PROP_NAME =prop_name
            prop_values = data[PROP_NAME]#'pair_lj_energy']
            time_steps = data['timestep']
            pe = data['potential_energy']
            print(job)
            len_prof = len(job.sp.quench_temp_prof)
            colors = plt.cm.plasma(np.linspace(1,0,len_prof/2))
            for i in range(0,len_prof,2):
                current_point = job.sp.quench_temp_prof[i]
                next_point = job.sp.quench_temp_prof[i+1]
                start_time = current_point[0]
                end_time = next_point[0]
                if current_point[1]!=next_point[1]:
                    print('WARNING! Detected a non isothermal step')
                target_T = current_point[1]
                #print(start_time,end_time)
                #print(time_steps)
                if np.isclose(target_T,filter_temp,rtol=rtol) or show_all: 
                    #print(time_steps)
                    #print(start_time,end_time)
                    indices = np.where((time_steps>=start_time)&(time_steps<=end_time))
                    #print(indices)
                    start_index = indices[0][0]
                    end_index = indices[0][-1]
                    #print('start_index',start_index,'end_index',end_index)
                    #print('start_index',start_index,'end_index',end_index)
                    sliced_ts = time_steps[start_index:end_index+1]
                    sliced_prop_vals = prop_values[start_index:end_index+1]
                    sliced_pe = pe[start_index:end_index+1]
                    #print(sliced_ts)
                    #print(sliced_prop_vals)
                    label = 'T:{}'.format(target_T)
                    #print(i/2)
                    plot_this(job,
                              sliced_ts,
                              sliced_prop_vals,
                              sliced_pe,
                              colors[int(i/2)],
                              label,
                              normalize_by_mean=normalize_by_mean)

                    
def line_intersect(m1, b1, m2, b2):
    if m1 == m2:
        print ("These lines are parallel!!!")
        return None
    # y = mx + b
    # Set both lines equal to find the intersection point in the x direction
    # m1 * x + b1 = m2 * x + b2
    # m1 * x - m2 * x = b2 - b1
    # x * (m1 - m2) = b2 - b1
    # x = (b2 - b1) / (m1 - m2)
    x = (b2 - b1) / (m1 - m2)
    # Now solve for y -- use either line, because they are equal here
    # y = mx + b
    y = m1 * x + b1
    return x,y   

def find_Tg(quenchTs, mean_vals):
    if False:#sap<=50.:
        use_first_deviation = True
        if use_first_deviation:
            model = piecewise(quenchTs, mean_vals)
            if len(model.segments) == 2:
                lines = []
                l1 = model.segments[0]
                m1 = l1.coeffs[1]
                b1 = l1.coeffs[0]
                l2 = model.segments[1]
                m2 = l2.coeffs[1]
                b2 = l2.coeffs[0]
            f = InterpolatedUnivariateSpline(quenchTs, mean_vals, k=2)
            dxdT = f.derivative(n=1)
            dx_dTs = dxdT(quenchTs)
            dev_index = np.where(np.abs(dx_dTs)>m1)[0][0]
            x=quenchTs[dev_index]
            y=mean_vals[dev_index]
        else:
            print('using derivatives')
            f = InterpolatedUnivariateSpline(quenchTs, mean_vals, k=2)
            dxdT = f.derivative(n=1)
            d2xdT = f.derivative(n=2)
            dx_dTs = dxdT(quenchTs)
            d2x_dT2s = d2xdT(quenchTs)
            max_dx2 = np.max(d2x_dT2s)
            min_dx2 = np.min(d2x_dT2s)
            max_i = np.where(d2x_dT2s==max_dx2)[0][0]
            min_i = np.where(d2x_dT2s==min_dx2)[0][0]
            x = (quenchTs[min_i]+quenchTs[max_i])/2
            y = (mean_vals[min_i]+mean_vals[max_i])/2
    else:
        print('using line iftting')
        #plot_data_with_regression(quenchTs, mean_vals)
        model = piecewise(quenchTs, mean_vals)
        #print(model) 
        if len(model.segments) == 2:
            lines = []
            l1 = model.segments[0]
            m1 = l1.coeffs[1]
            b1 = l1.coeffs[0]
            l2 = model.segments[1]
            m2 = l2.coeffs[1]
            b2 = l2.coeffs[0]
            x,y = line_intersect(m1,b1,m2,b2)
            
        else:
            print('WARNING: found {} line segments in regression!Expecting 2'.format(len(model.segments)))
    return x,y

def Fit_Diffusivity1(Ts,
                    Ds,
                    method='use_viscous_region',
                    min_D=1e-8,
                    ver=1,
                    viscous_line_index=1,
                    l1_T_bounds=[0,1],
                    l2_T_bounds=[0,1]):
    indices = np.where(Ds>min_D)#0.00000095)
    print("in common, indices:",indices)
    print("00", indices[0][0])
    start_index = indices[0][0]
    D_As=Ds[start_index:]
    quenchTs=Ts[start_index:]
    #print('quenchTs',quenchTs)
    model = piecewise(quenchTs, D_As)
    #print(ver)
    if ver==4:
        #print('ver 4')
        line_vals=[]
        Ts_low_i = np.where(Ts>=l1_T_bounds[0])[0]
        if len(Ts_low_i)==0:
            raise ValueError('lower bound for T fitting of line 1 too low. Use a higher T')
        l1_low_i = Ts_low_i[0]
        Ts_low_i = np.where(Ts>=l2_T_bounds[0])[0]
        if len(Ts_low_i)==0:
            raise ValueError('lower bound for T fitting of line 2 too low. Use a higher T')
        l2_low_i = Ts_low_i[0]
        
        Ts_high_i = np.where(Ts<=l1_T_bounds[1])[0]
        if len(Ts_high_i)==0:
            raise ValueError('upper bound for T fitting of line 1 too high. Use a lower T')
        l1_high_i = Ts_high_i[-1]
        Ts_high_i = np.where(Ts<=l2_T_bounds[1])[0]
        if len(Ts_high_i)==0:
            raise ValueError('upper bound for T fitting of line 2 too high. Use a lower T')
        l2_high_i = Ts_high_i[-1]
        #print('Ts_high_i',Ts_high_i)
        l1Ts=Ts[l1_low_i:l1_high_i+1]
        l1Ds=Ds[l1_low_i:l1_high_i+1]
        #print(l1_low_i,l1_high_i,l1Ts)
        l2Ts=Ts[l2_low_i:l2_high_i+1]
        l2Ds=Ds[l2_low_i:l2_high_i+1]
        #print(l2_low_i,l2_high_i,l2Ts,'Ts',Ts)
        par = np.polyfit(l1Ts, l1Ds, 1, full=True)
        m1 = par[0][0]#0-slope, 1-intercept
        b1 = par[0][1]
        xs = np.linspace(l1Ts[0],l1Ts[-1])
        ys = m1*xs+b1
        line_vals.append((xs,ys))
        
        par = np.polyfit(l2Ts, l2Ds, 1, full=True)
        m2 = par[0][0]#0-slope, 1-intercept
        b2 = par[0][1]
        xs = np.linspace(l2Ts[0],l2Ts[-1])
        ys = m2*xs+b2
        line_vals.append((xs,ys))
        
        x,y = line_intersect(m1,b1,m2,b2)
        Tg=x
        Tg_prop = y
        
        return Tg,Tg_prop,line_vals
    elif ver==3:
        line_vals=[]
        Ts_low_i = np.where(Ts>=l1_T_bounds[0])[0]
        if len(Ts_low_i)==0:
            raise ValueError('lower bound for T fitting of line 1 too low. Use a higher T')
        l1_low_i = Ts_low_i[0]
        Ts_low_i = np.where(Ts>=l2_T_bounds[0])[0]
        if len(Ts_low_i)==0:
            raise ValueError('lower bound for T fitting of line 2 too low. Use a higher T')
        l2_low_i = Ts_low_i[0]
        
        Ts_high_i = np.where(Ts<=l1_T_bounds[1])[0]
        if len(Ts_high_i)==0:
            raise ValueError('upper bound for T fitting of line 1 too high. Use a lower T')
        l1_high_i = Ts_high_i[-1]
        Ts_high_i = np.where(Ts<=l2_T_bounds[1])[0]
        if len(Ts_high_i)==0:
            raise ValueError('upper bound for T fitting of line 2 too high. Use a lower T')
        print('Ts_high_i',Ts_high_i)
        l2_high_i = Ts_high_i[-1]
        
        l1Ts=Ts[l1_low_i:l1_high_i]
        l1Ds=Ds[l1_low_i:l1_high_i]
        print(l1_low_i,l1_high_i,l1Ts)
        l2Ts=Ts[l2_low_i:l2_high_i]
        l2Ds=Ds[l2_low_i:l2_high_i]
        print(l2_low_i,l2_high_i,l2Ts)
        par = np.polyfit(l1Ts, l1Ds, 1, full=True)
        m1 = par[0][0]#0-slope, 1-intercept
        b1 = par[0][1]
        xs = np.linspace(l1Ts[0],l1Ts[-1])
        ys = m1*xs+b1
        line_vals.append((xs,ys))
        
        par = np.polyfit(l2Ts, l2Ds, 1, full=True)
        m2 = par[0][0]#0-slope, 1-intercept
        b2 = par[0][1]
        xs = np.linspace(l2Ts[0],l2Ts[-1])
        ys = m2*xs+b2
        line_vals.append((xs,ys))
        if viscous_line_index==0:
            Tg = -b1/m1
            Tg_prop = 0.
        elif viscous_line_index==1:
            Tg = -b2/m2
            Tg_prop = 0.
        else:
            x,y = line_intersect(m1,b1,m2,b2)
            Tg=x
            Tg_prop = y

        return Tg,Tg_prop,line_vals
    elif ver==2:
        n_lines=len(model.segments)
        if n_lines == 0:
            raise ValueError('Found zero lines in piecewise fitting')
        lines=[]
        line_vals=[]
        for i in range(n_lines):
            line = model.segments[i]
            lines.append(line)
            xs = np.linspace(line.start_t,line.end_t)
            ys = line.coeffs[1]*xs+line.coeffs[0]
            line_vals.append((xs,ys))
         
        if method=='use_viscous_region':
            if n_lines>1:
                l2=lines[viscous_line_index]
            else:
                l2=lines[0]
            m2 = l2.coeffs[1]
            b2 = l2.coeffs[0]
            Tg = -b2/m2
            Tg_prop = 0.
        else:
            Tg,Tg_prop=find_Tg(mean_vals=Ds,quenchTs=Ts)
        return Tg,Tg_prop,line_vals
    elif ver==1:
        if len(model.segments) == 2:
            l1 = model.segments[0]
            m1 = l1.coeffs[1]
            b1 = l1.coeffs[0]
            l2 = model.segments[1]
            m2 = l2.coeffs[1]
            b2 = l2.coeffs[0]
            x,y = line_intersect(m1,b1,m2,b2)
            xs1 = np.linspace(l1.start_t,l1.end_t)#np.linspace(l1.start_t,(x+(l1.end_t-l1.start_t)*0.2))
            ys1 = l1.coeffs[1]*xs1+l1.coeffs[0]
            xs2 = np.linspace(l2.start_t,l2.end_t)#np.linspace((x-(l2.end_t-l2.start_t)*0.2),l2.end_t)
            ys2 = l2.coeffs[1]*xs2+l2.coeffs[0]
            
            if method=='use_viscous_region':
                Tg = -b2/m2
                Tg_prop = 0.
            elif method == 'intersection':
                Tg=x
                Tg_prop=y
        else:
            print('WARNING: found {} line segments in regression!'.format(len(model.segments)))
        
        return Tg,Tg_prop,xs1,ys1,xs2,ys2
        
        
def Calc_Diffusivity(eq_time,
                     eq_msd,
                     fit_method='curve_fit'):
    #fit_method='curve_fit'#'power_law','poly_fit'
    if fit_method=='curve_fit':
        norm_eq_time = (eq_time-eq_time[0])
        #print(norm_eq_time,eq_msd)
        popt, pcov = curve_fit(lambda t,m,b: m*t+b ,
                                       eq_time,
                                       eq_msd,
                                       p0=[1.,0.0],
                                       bounds=([-1,0.0],[np.infty,np.infty]))
        drdt_A = popt[0]
        m=popt[0]
        b=popt[1]
    elif fit_method=='poly_fit':
        par = np.polyfit(time, msd, 1, full=True)
        drdt_A = par[0][0]#0-slope, 1-intercept
        m=par[0][0]
        b=par[0][1]
    elif fit_method=='power_law':
        popt, pcov = curve_fit(lambda t,w,x1: (w*t)**x1 ,
                           time,
                           msd,
                           p0=[0.2,1.0],
                           #p0=[1.0],
                           #bounds=([-np.infty,-np.infty],[np.infty,np.infty])
                           #bounds=([0],[4.0]))
                           maxfev=2000000,
                           bounds=([0.0,0.0],[1.0,4.0]))
        raise NotImplementedError('Diffusivity not determined')

    #calculate the diffusion coefficient
    dimensions=3
    D = drdt_A/(2*dimensions)
    return D,m,b

def getDiffusivities(project,df_curing,sortby='quench_T',name='bparticles',quench_time=1e7,use_first_trial=True):
    """
    returns diffusivity in units of D^2/tau where D and tau are distance and time units.
    Note that time is not in time steps.
    """
    Ts=[]
    Ds=[]
    for key,df_grp in df_curing.groupby('cooling_method'):
        if key=='quench' and quench_time is not None:
            df_filt = df_grp[(df_grp.quench_time==quench_time)]
        else:
            df_filt = df_grp
        df_sorted=df_filt.sort_values(sortby)
        for q_T,q_T_grp in df_sorted.groupby('quench_T'):
            for job_id in q_T_grp.index:
                job = project.open_job(id=job_id)
                if job.isfile('msd.log'):
                    log_path = job.fn('msd.log')
                    data = np.genfromtxt(log_path, names=True)
                    prop_values = data[name]#'pair_lj_energy']
                    equilibriated_ts_percentage = 0.5
                    if key=='anneal':
                        times,msds,qTs = get_split_quench_job_msd(job,name)
                        for j,msd in enumerate(msds):
                            start_index = int(len(times[j])*equilibriated_ts_percentage)
                            time=times[j]*job.sp.md_dt
                            quench_T = qTs[j]
                            eq_msd = msd[start_index:]
                            eq_time = time[start_index:]
                            D_A,m,b = Calc_Diffusivity(eq_time,eq_msd,'curve_fit')
                            Ts.append(quench_T)
                            Ds.append(D_A)
                    else:
                        all_time_steps = data['timestep']
                        start_index = int(len(all_time_steps)*equilibriated_ts_percentage)
                        time=all_time_steps*job.sp.md_dt
                        quench_T = job.sp.quench_T
                        eq_msd = prop_values[start_index:]
                        eq_time = time[start_index:]
                        #print(job)
                        D_A,m,b = Calc_Diffusivity(eq_time,eq_msd,'curve_fit')
                        Ts.append(quench_T)
                        Ds.append(D_A)
                    if use_first_trial:
                        break#just using the first data point in this quench_T instead of mean
    Ts=np.asarray(Ts)
    Ds=np.asarray(Ds)
    return Ts,Ds

def savefig(plt,nbname,figname,transparent=True):
    import os
    if not os.path.exists(nbname):
        os.makedirs(nbname)
    plt.savefig(os.path.join(nbname,figname),transparent=transparent)