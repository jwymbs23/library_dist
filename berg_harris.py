import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def norm_test():
    rand_list = np.random.normal(0,1,600)
    rand_list = np.concatenate((rand_list,np.random.normal(5,1,600)))
    rand_list = np.concatenate((rand_list,np.random.normal(7,0.4,600)))
    sort_prob_list = sorted(rand_list)
    n_points = len(rand_list)
    max_fourier = 100
    cdf = make_cdf(n_points)
    x = np.linspace(sort_prob_list[0],sort_prob_list[-1],n_points)
    plt.plot(sort_prob_list,cdf)
    #plt.plot(x,rand_list)
    plt.show()
    #f_coeffs, interval_list, interval_vals = sub_interval(sort_prob_list,cdf,max_fourier)
    #interpolate(x,interval_list,interval_vals,f_coeffs)
    F_est, xbounds,m,di = integrate_fourier(n_points,cdf,max_fourier,sort_prob_list)

    x_resolution = 0.001
    x_vals = np.linspace(xbounds[0], xbounds[-1], int(1/x_resolution))
    F = analytic_fit(x_vals,xbounds,m,di, x_resolution)
#    dF = derivative(F,x_vals,x_resolution,z,x_bounds[-1])

    plot(len(x_vals),x_vals,F,cdf,sort_prob_list,m)



def analytic_fit(x_vals ,x_bounds, m, di, x_resolution):
#    print(m,di,x_bounds)
    #print(x_bounds[0]/x_resolution)
    x_range = x_bounds[1] - x_bounds[0]
    F = np.asarray([(x - x_bounds[0]) / (x_range) for x in x_vals])
    for mm in range(1,m+1):
        for cx,x in enumerate(x_vals):
            x_range_part = (x - x_bounds[0])
            F[cx] += di[mm]*np.sin(mm*np.pi/x_range * x_range_part)
    for cx,x in enumerate(x_vals):
        if x < x_bounds[0]:
            F[cx] = 0
    return F


def interpolate(x_vals ,interval_list, interval_vals,f_coeffs):
#    print(f_coeffs,interval_list,interval_vals)
    n_points = len(x_vals)
    del_x = x_vals[1] - x_vals[0]
    x_vals = np.concatenate((x_vals,[x_vals[-1]+del_x]))
    F = []
    xv = []
    for i in range(len(interval_list)):
        F_temp = []
        x_vals_temp = []
        x_range = interval_vals[i][1]  - interval_vals[i][0]
        #        F_au = interval_list[i][0]/interval_list[-1][1]
        #print(F_au)
        x_low  = int((interval_vals[i][0] - interval_vals[0][0])/(interval_vals[-1][1] - interval_vals[0][0])*n_points)
        x_high = int((interval_vals[i][1] - interval_vals[0][0])/(interval_vals[-1][1] - interval_vals[0][0])*n_points)
        F_au = interval_list[i][0]/n_points
#        print(F_au)
        for x in range(x_high - x_low+1):
            F_temp.append(F_au + (interval_list[i][1] - interval_list[i][0])/interval_list[-1][1]  *  (x_vals[x+x_low] - interval_vals[i][0]) / (x_range))
            x_vals_temp.append(x_vals[x+x_low])
            #F[x] = F_au + (interval_list[i][1] - interval_list[i][0])/interval_list[-1][1]  *  (x_vals[x] - interval_vals[i][0]) / (x_range)
#        plt.plot(x_vals,F)
#        plt.show()
        for mm in range(1,len(f_coeffs[i])):
#            print(f_coeffs[i])
            for cx in range(x_high - x_low+1):#enumerate(x_vals):
                x_range_part = (x_vals[cx+x_low] - interval_vals[i][0])
                F_temp[cx] += f_coeffs[i][mm]*np.sin(mm*np.pi/x_range * x_range_part) * x_range/(interval_vals[-1][1] - interval_vals[0][0])
                #F[cx] += f_coeffs[i][mm]*np.sin(mm*np.pi/x_range * x_range_part) * x_range/(interval_vals[-1][1] - interval_vals[0][0])
        F.append(F_temp)
        xv.append(x_vals_temp)
#    print(xv[0])
    shift = np.abs(int(xv[0][0]/del_x))
    #shift = 0
    F[0].pop(0)
    xv[0].pop(0)

    xv.insert(0,[i*del_x for i in range(shift+2)])#np.linspace(0,xv[0],shift),xv))
    initial_zeros = np.zeros(shift)
    #F = np.asarray(F)
    #x_vals = np.concatenate((initial_zeros,x_vals))
    F.insert(0,[0 for i in range(shift+2)])
    print(F[0])
    dF_interval = []
    xv_interval = []
    for i in range(len(F)):
        dF_temp = []
        xv_temp = []
        print(i)
        for j in range(len(F[i])-1):
            dF_temp.append((F[i][j+1] - F[i][j])/(xv[i][j+1] - xv[i][j]))
            xv_temp.append(xv[i][j])
        dF_interval.append(dF_temp)
        xv_interval.append(xv_temp)

    dF,x_vals = patch(dF_interval,xv_interval,interval_list,interval_vals, shift,n_points)    
    print(n_points+shift+2, len(x_vals))
    x_vals = [i*((n_points+shift+3)/len(x_vals)) for i in x_vals]
    x = np.asarray(x_vals[shift+1:])
    plt.plot(x_vals[shift+1:],dF[shift+1:])
    plt.plot(x, 1/3*(1/np.sqrt(2*np.pi)*np.exp(-x**2/2) + 1/np.sqrt(2*np.pi)*np.exp(-(x - 10)**2/2) + 1/(0.4*np.sqrt(2*np.pi))*np.exp(-((x - 15)/0.4)**2/2)) )
    plt.show()
    return dF,x_vals,len(interval_list)


def patch(F, x_vals, interval_list, interval_vals,shift,points_in_interval):
    indices = zip([i for i in range(len(F))],[i+1 for i in range(len(F)-1)])
    del_x = x_vals[0][1] - x_vals[0][0]
    width=100
    patch_width = width*del_x
    F_full = []
    x_vals_full = []
    for l_window,u_window in indices:
        jump = (F[u_window][0] - F[l_window][-1])/del_x/width
        curve_l = np.sign(F[l_window][-1] - F[l_window][-2])
        if curve_l == 0:
            curve_l = 1
        curve_u = np.sign(F[u_window][1] - F[u_window][0])
        print(curve_l, curve_u)

        curve_l = 1
        curve_u = 1
        for i in range(1,width):
            F[l_window][-i] += 0*curve_l*jump/(2*patch_width) * (x_vals[l_window][-i] - (x_vals[l_window][-1] - patch_width))**2
            F[u_window][i-1]-= 0*curve_u*jump/(2*patch_width) * (x_vals[u_window][i-1]- (x_vals[u_window][0]  + patch_width))**2
            #print(F[u_window][0])
        F[u_window].pop(0)
        x_vals[u_window].pop(0)
    x_vals_full = [j-ci*del_x for ci,i in enumerate(x_vals) for j in i]
    F_full = [j for i in F for j in i]
    return F_full, x_vals_full


def custom_dist(sample_prob_dist):
    sort_prob_list = sorted(sample_prob_dist)
    n_points = len(sort_prob_list)
    max_fourier = 1000
    cdf = make_cdf(n_points)
    F_est, x_bounds, f_num, f_coeffs = integrate_fourier(n_points,cdf,max_fourier,sort_prob_list)
    #print(f_num,F_est)
    #plot(n_points,sort_prob_list,F_est,cdf)
    Func = [(F_est[i+1] - F_est[i])/(sort_prob_list[i+1] - sort_prob_list[i]) for i in range(n_points - 1)]
    #print(Func,sort_prob_list)
    Func.insert(0,0)
    return Func, x_bounds, f_num, f_coeffs
    


def custom_dist_interval(sample_prob_dist):
    sort_prob_list = sorted(sample_prob_dist)
    n_points = len(sort_prob_list)
    max_fourier = 4
    cdf = make_cdf(n_points)
    f_coeffs, interval_list, interval_vals = sub_interval(sort_prob_list,cdf,max_fourier)
    x = np.linspace(sort_prob_list[0],sort_prob_list[-1],n_points)
    dF, x_vals,n_intervals = interpolate(x,interval_list,interval_vals,f_coeffs)
    #    F_est, x_bounds, f_num, f_coeffs = integrate_fourier(n_points,cdf,max_fourier,sort_prob_list)
    #print(f_num,F_est)
    #plot(n_points,sort_prob_list,F_est,cdf)
    #Func = [(F_est[i+1] - F_est[i])/(sort_prob_list[i+1] - sort_prob_list[i]) for i in range(n_points - 1)]
    #print(Func,sort_prob_list)
    #Func.insert(0,0)
    return dF, x_vals,n_intervals#Func, x_bounds, f_num, f_coeffs


def make_cdf(n_points):
    cdf = []
    for i in range(n_points):
        cdf.append(i/n_points)#cdf[i - 1] + sort_rand_list[i])
    return cdf


def sub_interval(sort_prob_list,cdf,max_fourier):
    interval_list = [[0,len(sort_prob_list)-1]]
    interval_vals = [[sort_prob_list[0],sort_prob_list[-1]]]
    f_coeffs = []
    f_coeffs,interval_list,interval_vals = integrate_fourier_sub(cdf,max_fourier,sort_prob_list,0,interval_list,f_coeffs,interval_vals)
    return f_coeffs, interval_list,interval_vals
    

def integrate_fourier_sub(cdf,max_fourier,sort_prob_list,current_interval, interval_list,f_coeffs,interval_vals):
#    print(current_interval, len(interval_list))
    if current_interval >= len(interval_list):
        return current_interval, f_coeffs, interval_list,interval_vals
#    print(current_interval, interval_list, len(sort_prob_list))
    sort_prob_list_interval = sort_prob_list[interval_list[current_interval][0]:interval_list[current_interval][1]]
    n_points = len(sort_prob_list_interval)
#    print(n_points)
#    print(interval_list)
    cdf_interval = make_cdf(n_points)#cdf[interval_list[current_interval][0]:interval_list[current_interval][1]]
    F_est = np.zeros(n_points)
    F_0 = np.zeros(n_points)
    #di = np.zeros(max_fourier+1)
    di = [0]
#    print(sort_prob_list_interval)
    x_range = sort_prob_list_interval[-1] - sort_prob_list_interval[0]
    for i in range(n_points):
        F_0[i] = (sort_prob_list_interval[i] - sort_prob_list_interval[0])/x_range
        F_est[i] = (sort_prob_list_interval[i] - sort_prob_list_interval[0])/ x_range
        #    plt.plot(sort_prob_list,F_est)
        #    plt.show()
    for m in range(1,max_fourier+1):
        di.append(0)
        for i in range(n_points - 1):
            xl = (sort_prob_list_interval[i] - sort_prob_list_interval[0])/ x_range
            xr = (sort_prob_list_interval[i+1] - sort_prob_list_interval[0])/ x_range
            #fourier component
            di[m] += (i/n_points - xl)*np.cos(m*np.pi * xl)/(m*np.pi) + np.sin(m*np.pi * xl)/(m*m*np.pi*np.pi)
            di[m] -= (i/n_points - xr)*np.cos(m*np.pi * xr)/(m*np.pi) + np.sin(m*np.pi * xr)/(m*m*np.pi*np.pi)
        di[m] *= 2
#        print(di)
        for j in range(n_points):
            xrange_part = sort_prob_list_interval[j] - sort_prob_list_interval[0]
            F_est[j] += di[m]*np.sin(m*np.pi/x_range*xrange_part)
            #        print(np.sqrt(-0.5*np.log(0.0001/2)) * np.sqrt((len(F_est) + len(cdf))/(len(F_est)*len(cdf))))
        ks_test = ks_2samp(F_est,cdf_interval)
        #print(m, ks_test)
        #print(m,max_fourier)
        if m == max_fourier:
            print(ks_test[1])
            p_cutoff = 1 - 100/(interval_list[current_interval][1] - interval_list[current_interval][0])
            print('p cutoff', p_cutoff, ks_test[1])
            if ks_test[1] < p_cutoff:
                print('hi')
                print(ks_test)
                diff_array = [abs(l-k) for l,k in zip(F_est,cdf_interval)]
#                plt.plot(F_est)
#                plt.plot(cdf_interval)
#                plt.show()
                max_diff_index = diff_array.index(max(diff_array)) + interval_list[current_interval][0]
                if max_diff_index - interval_list[current_interval][0] > 100 and max_diff_index - interval_list[current_interval][1] < -100:
                    print('split at ', max_diff_index)
                    old_interval = interval_list[current_interval]
                    
                    print(interval_list)
                    interval_list.insert(current_interval,[old_interval[0], max_diff_index])
                    interval_list.insert(current_interval+1, [max_diff_index, old_interval[1]])
                    interval_list.pop(current_interval+2)
                    
                    interval_vals.insert(current_interval,[sort_prob_list[interval_list[current_interval][0]], sort_prob_list[max_diff_index]])
                    interval_vals.insert(current_interval+1, [sort_prob_list[max_diff_index], sort_prob_list[old_interval[1]]])
                    interval_vals.pop(current_interval+2)
                    print(interval_list)
                    
                    #current_interval += 1
                    integrate_fourier_sub(cdf,max_fourier,sort_prob_list,current_interval,interval_list,f_coeffs,interval_vals)
                else:
                    f_coeffs.append(di)
                    current_interval += 1
                    print(current_interval, len(interval_list))
                    if current_interval == len(interval_list):
                        return f_coeffs, interval_list,interval_vals
                    else:
                        integrate_fourier_sub(cdf,max_fourier,sort_prob_list,current_interval,interval_list,f_coeffs,interval_vals)
            else:
                f_coeffs.append(di)
                current_interval += 1
                print(current_interval, len(interval_list))
                if current_interval == len(interval_list):
                    return f_coeffs, interval_list,interval_vals
                else:
                    integrate_fourier_sub(cdf,max_fourier,sort_prob_list,current_interval,interval_list,f_coeffs,interval_vals)
                    #                return [F_est, [sort_prob_list[0], sort_prob_list[-1]], m, di]
    return f_coeffs, interval_list,interval_vals
    print('Unable to converge')
    exit(0)






def integrate_fourier(n_points,cdf,max_fourier,sort_prob_list):
    F_est = np.zeros(n_points)
    F_0 = np.zeros(n_points)
    di = np.zeros(max_fourier)
    x_range = sort_prob_list[-1] - sort_prob_list[0]
    for i in range(n_points):
        F_0[i] = (sort_prob_list[i] - sort_prob_list[0])/x_range
        F_est[i] = (sort_prob_list[i] - sort_prob_list[0])/ x_range
        #    plt.plot(sort_prob_list,F_est)
        #    plt.show()
    for m in range(1,max_fourier):
        plt.clf()
        x_resolution = 0.001
        x_vals = np.linspace(sort_prob_list[0], sort_prob_list[-1], int(1/x_resolution))
        F = analytic_fit(x_vals,[sort_prob_list[0],sort_prob_list[-1]],m,di, x_resolution)
        plot(len(x_vals),x_vals,F,cdf,sort_prob_list,m)
        for i in range(n_points - 1):
            xl = (sort_prob_list[i] - sort_prob_list[0])/ x_range
            xr = (sort_prob_list[i+1] - sort_prob_list[0])/ x_range
            #fourier component
            di[m] += (i/n_points - xl)*np.cos(m*np.pi * xl)/(m*np.pi) + np.sin(m*np.pi * xl)/(m*m*np.pi*np.pi)
            di[m] -= (i/n_points - xr)*np.cos(m*np.pi * xr)/(m*np.pi) + np.sin(m*np.pi * xr)/(m*m*np.pi*np.pi)
        di[m] *= 2
#        print(di)
        for j in range(n_points):
            xrange_part = sort_prob_list[j] - sort_prob_list[0]
            F_est[j] += di[m]*np.sin(m*np.pi/x_range*xrange_part)
            #        print(np.sqrt(-0.5*np.log(0.0001/2)) * np.sqrt((len(F_est) + len(cdf))/(len(F_est)*len(cdf))))
        ks_test = ks_2samp(F_est,cdf)
        print(m, ks_test)
        if ks_test[1] > 0.9999999:#1.95 * np.sqrt((n_points + len(F_0))/ (n_points*len(F_0))) :
            exit(0)
            print(ks_test)
            return F_est, [sort_prob_list[0], sort_prob_list[-1]], m, di
    return F_est, [sort_prob_list[0], sort_prob_list[-1]], m, di
    print('exceeded max fourier components')
    exit(0)
    #return F_est,'fail'

#def make_analytic():



def plot(n_points,sort_prob_list,F_est,cdf,hist_vals,m):
    axes = plt.gca()
    axes.set_ylim([0,0.4])

    Func = [(F_est[i+1] - F_est[i])/(sort_prob_list[i+1] - sort_prob_list[i]) for i in range(n_points - 1)]
#    print(Func,sort_prob_list)
    Func.insert(0,0)
    x = np.linspace(sort_prob_list[0],sort_prob_list[-1],n_points)
    #plt.plot(sort_prob_list,cdf)
    #plt.plot(sort_prob_list,F_est)
    #plt.show()
    #plt.plot(sort_prob_list,F_est)
    plt.plot(sort_prob_list[1:],Func[1:], color = 'red',label = 'Berg-Harris Fit')
    hist,bins,p = plt.hist(hist_vals, 100,normed = True, color = 'grey', alpha = 0.6,label='Histogram')
    plt.plot(x,1/3*(1/np.sqrt(2*np.pi)*np.exp(-x**2/2) + 1/np.sqrt(2*np.pi)*np.exp(-(x - 5)**2/2) + 1/(np.sqrt(2*np.pi)*0.4)*np.exp(-((x - 7)/0.4)**2/2)) ,color = 'black', label = 'Correct Distribution')
    #1/np.sqrt(2*np.pi)*np.exp(-x**2/2))
    #plt.plot(sort_prob_list,ff)
    plt.legend()
    s = "Fourier Components: " + str(m-1)
    plt.text(0,0.24,s)
    plt.savefig('berg_harris_%02d.png'%(m), dpi=300)



#print(norm_test())
#custom_dist(np.random.normal(0,1,2000))
