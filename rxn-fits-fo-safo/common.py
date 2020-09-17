import numpy as np
import math
from scipy.optimize import curve_fit
import sys,traceback

def f_t(times,C,H,Ea,kT,a_start,a_inf,breakAt_a=None,model='FO'):
    alphas = []
    the_times = []
    alpha=a_start
    #print(H)
    #a_inf = 0.96
    try:
        for t in times:
            k = H*math.exp(-Ea/(kT))
            if model == 'SAFO':
                dadt = k*(a_inf-alpha)*(1+C*alpha)
            elif model == 'FO':
                dadt = k*(a_inf-alpha)
            elif model == 'SO':
                dadt = k*(a_inf-alpha)**2
            elif model == 'SASO':
                dadt = k*(1-alpha)*(a_inf-alpha)*(1+C*alpha)
            alpha += dadt
            alphas.append(alpha)
            the_times.append(t)

            if (breakAt_a is not None) and (alpha >= breakAt_a):
                t_minutes = t/60
                print('{} reached @ {} minutes according to the model'.format(alpha,t_minutes))
                return alphas,minutes,t_minutes
                #    print('done at',t)
                #    break
    except Exception as e:
        print('math.exp(-Ea/(kT))',math.exp(-Ea/(kT)))
        print('H',H)
        plt.plot(the_times,alphas,marker='+')
        print(the_times,alphas)
        raise e
    return alphas
    
def fit_curing_profile_with_model(job,model,print_error=False):
    #print('fitting job',job)
    bond_percent_index = 9
    #C=8.12 #Temperature independent acceleration constant. Aldridge, M., Wineman, A., Waas, A. & Kieffer, J.  (2014).
    suggested_C = 8.12#1e-10#0.0
    C_tolerance=1e-5
    data = np.genfromtxt(job.fn('out.log'),names=True)
    bond_percents = data['bond_percentAB']
    time_steps = data['timestep']
    alpha_inf = job.sp.stop_after_percent
    
    truncated_cure_fractions = []
    truncated_time_steps = []
 
    # We cut off the cure profile after stop_after_percent because we want the cure profile to be realistic. 
    # After we stop bonding the cure profile is just flat and not realistic.
    last_index = next((i for i, v in enumerate(bond_percents) if v >=alpha_inf), -1)
    first_index = next((i for i, v in enumerate(bond_percents) if v >0), -1)
    if last_index<=0:#maybe the system did not cure till the desired cure percent. So just take the last cure of the profile
        last_index=len(bond_percents)
    if last_index > 0 and first_index >= 0 and (last_index-first_index)>1:
        truncated_time_steps.extend(time_steps[first_index:last_index]*job.sp.md_dt)#/0.01)
        truncated_cure_fractions.extend(bond_percents[first_index:last_index]/100.)
        Ea=job.sp.activation_energy#1.0
        a_inf = truncated_cure_fractions[-1]#0.96
        import warnings
        np.seterr(all='raise')
        plot_fit_fails=True
        label='Successfully fit the curing curve'
        success=False
        try:
            #print('math.exp(-Ea/(kT))',math.exp(-Ea/(kT)))
            popt, pcov = curve_fit(lambda times, H: f_t(times, 
                                                          suggested_C, 
                                                          H,
                                                          Ea,
                                                          job.sp.kT,
                                                          truncated_cure_fractions[0],
                                                          a_inf,model=model),
                               truncated_time_steps,truncated_cure_fractions,
                                   p0=[1e-4],
                                   maxfev=20000,
                                   bounds=([0.0],
                                           [np.infty]))
                                           #[np.infty,np.infty]))
                               #p0=[1e-4,1e-10],
                               #bounds=([0,1e-10],#[0,1e-10],#[suggested_C,1e-4],
                                # [np.infty,np.infty]))
            success=True
        
        except Exception as e:
            if print_error:
                print(e)
                traceback.print_exc(file=sys.stdout)
        #if print_error:
        #    print(error)
        if success:
            ydata = np.asarray(truncated_cure_fractions)
            #fit_ydata = f_t(truncated_time_steps,C,*popt,Ea,job.sp.kT,truncated_cure_fractions[0],a_inf,model=model)
            fit_ydata = f_t(truncated_time_steps,suggested_C,*popt,Ea,job.sp.kT,truncated_cure_fractions[0],a_inf,model=model)
            
            residuals = ydata - fit_ydata
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ydata-np.mean(ydata))**2)
            #print('ss_res',ss_res,'ss_tot',ss_tot)
            if ss_tot == 0:
                r_squared = 0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            C = suggested_C
            H = popt[0]
        else:
            r_squared=0.0
            C=suggested_C
            H=0.0
            truncated_cure_fractions=None
            fit_ydata=None
    else:
        success=False
        r_squared=0.0
        C=suggested_C
        H=0.0
        truncated_cure_fractions=None
        fit_ydata=None
        #print('Did not try to fit curing curves. last_index:',last_index,'first_index',first_index)
    return success,r_squared,C,H, truncated_time_steps,fit_ydata,first_index,last_index

def savefig(plt,nbname,figname,transparent=True):
    import os
    if not os.path.exists(nbname):
        os.makedirs(nbname)
    plt.savefig(os.path.join(nbname,figname),transparent=transparent)