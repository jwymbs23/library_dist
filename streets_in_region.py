import numpy as np
import sys, os
import googlemaps
from datetime import datetime
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
import berg_harris as bh
#address = "3000 Broadway, New York"

map_file = "Library.kml"
f = open(map_file, "r")
raw_map_data = f.readlines()
f.close()
segment_data = []
flag = 0
lib_count = 0
for ci,i in enumerate(raw_map_data):
    #get all street segments (later, add street names, street widths)
    if flag == 1:
        if "coordinates" in i:# == "</coordinates></LineString>\n":
            segment_data.append(lib_data)
            flag = 2
        if flag < 2:
            lib_data = [float(j) for j in i.strip('\n').split(',')]
            #print([float(j) for j in i.strip('\n').split(',')])
    if "coordinates" in i and flag < 2:
        lib_data = []
        flag = 1
    if flag == 2:
        lib_count += 1
        flag = 0
#    if ci > 500:
raw_map_data = []
#print(street_count)
lib_coords = np.asarray(segment_data)




boundary_file = 'boundary.kml'
b = open(boundary_file, "r")
raw_boundary = b.readlines()
b.close()
boundary_data = []
flag = 0
bound_count = 0
for ci,i in enumerate(raw_boundary):
    #get all street segments (later, add street names, street widths)
    if flag == 1:
        if "coordinates" in i:# == "</coordinates></LineString>\n":
            boundary_data.append(bound_data)
            flag = 2
        if flag < 2:
            bound_data.append([float(j) for j in i.strip('\n').split(',')])
            #print([float(j) for j in i.strip('\n').split(',')])
    if "coordinates" in i and flag < 2:
        bound_data = []
        flag = 1
    if flag == 2:
        bound_count += 1
        flag = 0
#    if ci > 500:

#print(boundary_data)



def isPointInPath(x, y, poly):
    num = len(poly)
    i = 0
    j = num - 1
    c = False
    for i in range(num):
        if  ((poly[i][1] > y) != (poly[j][1] > y)) and (x < (poly[j][0] - poly[i][0]) * (y - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]):
            c = not c
        j = i
    return c



dist_list = []
shell_width = 0.2
dist_cut = 10
inner_cut = 0.3
nbins = int(np.ceil(dist_cut/shell_width))
print(nbins)
count = [0 for i in range(nbins+1)]
count_no_scale = [0 for i in range(nbins+1)]
count_scaled = [0 for i in range(nbins+1)]

bin_edges = [i*shell_width for i in range(nbins+1)]


def miles_dist(p1, p2):
    R = 3961 #radius of the earth in miles
    dlon = (p1[0] - p2[0])*np.pi/180.
    dlat = (p1[1] - p2[1])*np.pi/180.
    #        dist = (dx*dx + dy*dy)
    #theta = np.arctan2(dlat,dlon)
    #angle_list.append(theta)
    a = (np.sin(dlat/2))**2 + np.cos(p1[1]*np.pi/180.) * np.cos(p2[1]*np.pi/180.) * (np.sin(dlon/2))**2
    c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )
    return R * c



NYC_area = 302.643 #land area of NYC 
max_diff = []
for i in range(0,len(lib_coords)-1):
    count_temp = [0 for i in range(nbins+1)]
    angle_list = []
    for j in range(i+1,len(lib_coords)):
        dist = miles_dist(lib_coords[i], lib_coords[j])
        #angle_list.append(theta)
        if dist < dist_cut and dist > inner_cut:
            count_temp[int(dist//shell_width)] += 1
            dist_list.append(dist)
    #angle_list.sort()
    #angle_list_shift = [angle_list[(i+1)%len(angle_list)] for i in range(0, len(angle_list))]
    #angle_list_shift[-1] += 2*np.pi
    ##print(angle_list_shift)
    ##print(zip(angle_list, angle_list_shift))
    #diff_list = [k - l for l, k in zip(angle_list, angle_list_shift)]
    #max_empty_angle = max(diff_list)
    #print(max_empty_angle)
    #max_diff.append(max_empty_angle)
    #    if max_empty_angle > 2.:
#    count_scaled = [(i[1])/(2*np.pi - max_empty_angle)*2*np.pi + i[0] for ci,i in enumerate(zip(count_scaled,count_temp))]
    #print(count_temp, count_scaled)
    #count = [sum(x) for x in zip(count, count_scaled)]
    #else:
    #    count = [sum(x) for x in zip(count, count_temp)]
    count_no_scale = [sum(x) for x in zip(count_no_scale, count_temp)]
    #print(sum(diff_list), max(diff_list))
#    print(diff_list)
#print(count_no_scale, count_scaled)
#hist, bins, p = plt.hist(max_diff, 50)
#plt.show()
#exit(0)


#dat,bins,p = plt.hist(dist_list,1000)
#norm = [dat[x]/x for x in range(1,1000)]
#plt.plot(bins[1:-1],norm)
#plt.show()



g_of_r = [0]
g_of_r_scaled = [0]
pmf = [0]
g_of_r_no_scale = [0]
pmf_no_scale = [0]
for i in range(1,len(bin_edges)):
#    g_of_r.append(count[i]/ ( 2*np.pi*(bin_edges[i] + 0.5*shell_width)*shell_width) / lib_count / (lib_count / NYC_area))
#    pmf.append(-np.log(g_of_r[-1])
    g_of_r_no_scale.append(2*count_no_scale[i]/( 2*np.pi*(bin_edges[i])*shell_width) * (NYC_area / lib_count) / lib_count)# + bin_edges[i]*0.1)
    #g_of_r_scaled.append(count_scaled[i]/( 2*np.pi*(bin_edges[i])*shell_width) * (NYC_area / lib_count) / lib_count)# + bin_edges[i]*0.06)
    pmf_no_scale.append(-np.log(g_of_r_no_scale[-1]))


def flatten_tail(ff,xx):
    #factor = (1 - g_of_r[-2])/(bin_edges[-2])
    decay = 2
    shift = 2.25
    factor = 0
    return [i + (1-i)*(1/(1 + np.exp(-decay * (xx[ci] - shift)))) for ci,i in enumerate(ff)], factor

#bin_edges[i] *(2*np.pi*shell_width) / (NYC_area/lib_count) * lib_count)    * factor 

#g_of_r.append(0)
#pmf.append(0)
#print(count)
#g_of_r_no_scale.append(0)
#pmf_no_scale.append(0)

def running_average(g_of_r_no_scale, pmf_no_scale):
    #running average:
    av_window = 1
    g_of_r_av = []
    pmf_av = []
    for i in range(len(g_of_r_no_scale) - av_window):
        g_of_r_av.append(np.mean(g_of_r_no_scale[i:i+av_window]))
        pmf_av.append(np.mean(pmf_no_scale[i:i+av_window]))
    bin_edges_av = [bin_edges[i+int(av_window*0.5)] for i in range(len(bin_edges) - av_window)]
    return g_of_r_av, pmf_av, bin_edges_av


#plt.plot(center, g_of_r2)#, align='center', width=width)
#plt.plot(bin_edges,g_of_r)
g_of_r = g_of_r_no_scale
#g_of_r,converge_factor = flatten_tail(g_of_r_no_scale, bin_edges)


#p_of_r = [i + converge_factor*bin_edges[ci]*((2*np.pi*bin_edges[ci]*shell_width) / (NYC_area/lib_count) * lib_count) for ci,i in enumerate(count_no_scale)]
#g_of_r_test = [0]
#for i in range(1,len(bin_edges)):
#    g_of_r_test.append(p_of_r[i]/( 2*np.pi*(bin_edges[i])*shell_width) * (NYC_area / lib_count) / lib_count)


g_of_r_av, pmf_av, bin_av = running_average(g_of_r,pmf_no_scale)

plt.plot(bin_av,g_of_r_av,linewidth=3,color='black',label="Radial Distribution Function")
plt.xlabel("r")
plt.ylabel("g(r)",rotation=0,labelpad=15)
plt.legend()
plt.tight_layout()
plt.savefig('g_of_r.png',dpi=600)
plt.show()
exit(0)

#print(bin_av, g_of_r_av)
#plt.plot(bin_edges,g_of_r_test)
#plt.plot(bin_edges,g_of_r)
#plt.plot(bin_av,g_of_r_av)
#plt.show()


#exit(0)
##plt.plot(bin_edges,pmf)
#plt.plot(bin_edges,pmf_no_scale)
#plt.plot(bin_edges_av, pmf_no_scale_av)
#
#xx = np.linspace(0,15,num=1000,endpoint=True)
#fit = np.polyfit(bin_edges[2:150], pmf_no_scale[2:150], 10)
#potential = np.poly1d(fit)
#plt.plot(xx,potential(xx))
##plt.show()


#vor = Voronoi(lib_coords)
#voronoi_plot_2d(vor)
#plt.show()




def resample_dists(dist_list):
    #get weights
    resampled = []
    weights = [1/(x-inner_cut*0.5) for x in dist_list]
    max_weight = max(weights)
    Pi = [i/max_weight for i in weights]
    z = 2*np.pi/(NYC_area)*np.mean(weights)#*len(weights))*sum(weights)
    pop = 0
    while pop < int(len(dist_list)*1):
        rand_ind = np.random.randint(0,len(dist_list))
        if np.random.rand() < Pi[rand_ind]:
            resampled.append(dist_list[rand_ind])
            pop += 1
#            if not pop%100:
#                print(pop, len(dist_list))
    return resampled, z



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


def derivative(F, x_vals, x_resolution,z,x_range):
    return [(F[i+1] - F[i])/(x_vals[1] - x_vals[0]) for i in range(len(F) - 1)]


resampled, z = resample_dists(dist_list)
dF_interval, x_vals_interval, n_intervals = bh.custom_dist_interval(resampled)

#dF_interval = [i/(2*np.pi*x_vals_interval[ci]*(x_vals_interval[1]-x_vals_interval[0])) for ci,i in enumerate(dF_interval)]
#plt.plot(x_vals_interval,dF_interval)
#plt.show()
x_step = x_vals_interval[1] - x_vals_interval[0]

smooth, x_bounds, m, di = bh.custom_dist(resampled)
#smooth = [i/(2*np.pi*x_vals_interval[ci]*(x_vals_interval[1]-x_vals_interval[0])) for ci,i in enumerate(dF_interval)]

x_resolution = 0.001
x_vals = np.linspace(0, x_bounds[1], int(1/x_resolution))
F = analytic_fit(x_vals,x_bounds,m,di, x_resolution)
dF = derivative(F,x_vals,x_resolution,z,x_bounds[-1])# - x_bounds[0])





def plot_hist_fit():
    dat,bins,p = plt.hist(resampled,100,normed=True,color='blue',alpha=0.3,label="Resampled Distance Histogram")
    plt.plot(x_vals[:-1],dF,color='green',linewidth=3,label="Berg-Harris")
    plt.plot(x_vals_interval, dF_interval,color='red',linewidth=3,label="Smoothed Piecewise Berg-Harris")
    plt.xlabel("r")
    plt.ylabel("P(r)",rotation=0,labelpad=15)
    plt.legend()
    plt.tight_layout()
    #plt.legend([line_up, line_down], ['Line Up', 'Line Down'])
    plt.savefig('hist_fit.png',dpi=600)
    plt.show()


plot_hist_fit()


#( 2*np.pi*(bin_edges[i])*shell_width) * (NYC_area / lib_count) / lib_count




#plt.show()


def flatten_distribution(dF_interval,dF,g_of_r_av,x_vals_interval,x_vals,bin_av):
    dF_interval = [i*max(g_of_r_av)/max(dF_interval) for i in dF_interval]
    flattendF_int = dF_interval

    flattendF_int, factor = flatten_tail(dF_interval,x_vals_interval)
#    flattendF_int = [i*1/flattendF_int[-1] for i in flattendF_int]

    dF = [i*max(g_of_r_av)/max(dF) for i in dF]
    flattendF = dF
    flattendF, factor = flatten_tail(dF,x_vals)
#    flattendF = [i*1/flattendF[-1] for i in flattendF]

    
    #plt.plot(x_vals[:-1],dF)
    plt.plot(x_vals_interval,flattendF_int,linewidth=3,color='red',label="Flattened Piecewise Berg-Harris")
    plt.plot(x_vals[:-1],flattendF,linewidth=3,color='green',label='Flattened Berg-Harris')
    #for i in dF:
    #    integ += i*(x_vals[1] - x_vals[0])
    #print(integ)
    
    #g_of_r_av = [i*1/g_of_r_av[-1] for i in g_of_r_av]
    plt.plot(bin_av,g_of_r_av,linewidth=3,color='black',label="Smoothed g(r)")
    plt.xlabel("r")
    plt.ylabel("g(r)",rotation=0,labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('g_of_r_fit.png',dpi=600)
    plt.show()
    return flattendF_int

flattendF = flatten_distribution(dF_interval,dF,g_of_r_av,x_vals_interval,x_vals,bin_av)







potential_actual = -10*np.log(flattendF)
potential = potential_actual
potential = [i if i < 30 else 30 for i in potential_actual]

#print(potential)
#plt.plot(x_vals[:-1], potential)
#plt.show()

from scipy.interpolate import interp1d


#'https://maps.googleapis.com/maps/api/directions/json?origin= &destination= & &key=YOUR_API_KEY'
#travel distance
#for i in range(len(lib_coords) - 1):
#    for j in range(i+1, len(lib_coords)):

#exit(0)




import glob
import pickle
import matplotlib.path as mpltPath
pickle_files = glob.glob('*.pkl')


from matplotlib.collections import LineCollection
N,S,W,E = 40.915568,40.495992,-74.257159,-73.699215
bounding_box = [W,E,S,N]#-74.256771,-73.699229,40.497721,40.915140]


def plot_voronoi():
    vor = Voronoi(lib_coords)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=(bounding_box[0],bounding_box[1]), ylim=(bounding_box[2], bounding_box[3]))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    #fig.subplots_adjust(left=-0.15, right=1.15, bottom=-0.15, top=1.15)
    coll = LineCollection(boundary_data, linewidths = 0.4,color = 'black')
    #rect = plt.Rectangle(bounding_box[::2],bounding_box[1] - bounding_box[0],bounding_box[3] - bounding_box[2],ec='none', lw=2, fc='none')
    #roadmap = np.asarray(seg_data)
    #plt.axis('off')
    #plt.savefig('roadmap_%s.png'%(model),dpi=1200)
    plt.plot(vor.vertices[:,0], vor.vertices[:, 1], 'ko', ms=1, color='red')
    for vpair in vor.ridge_vertices:
        if vpair[0] >= 0 and vpair[1] >= 0:
            v0 = vor.vertices[vpair[0]]
            v1 = vor.vertices[vpair[1]]
            # Draw a line from v0 to v1.
            plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=0.4,color='red')
    plt.scatter(lib_coords.T[0], lib_coords.T[1], s=5,color = 'green')
    #voro = voronoi_plot_2d(vor, ax=ax, pointsize=0.1, linewidths=0.1)
    ax.add_collection(coll)
    plt.tight_layout()
    plt.savefig('voronoi.png',dpi=600)
    plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=(bounding_box[0],bounding_box[1]), ylim=(bounding_box[2], bounding_box[3]))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
#fig.subplots_adjust(left=-0.15, right=1.15, bottom=-0.15, top=1.15)
coll = LineCollection(boundary_data, linewidths = 0.4,color = 'black')
ax.add_collection(coll)
plt.tight_layout()


grid_size = 0.002

x = np.arange(W,E,grid_size)#-74.257159, -73.699215, 0.1)
y = np.arange(S,N,grid_size)#40.495992, 40.915568, 0.1)

if 'nyc_grid.pkl' in pickle_files:
    in_poly_mat = pickle.load(open('nyc_grid.pkl','rb'))
    print(in_poly_mat)
else:
    in_poly_mat = np.asarray([ [0 for i in range(len(y))] for j in range(len(x))])
    print(in_poly_mat)
    flag = 0
    for i in range(len(x)):
        xc = x[i]
        for j in range(len(y)):
            yc = y[j]
            for b in boundary_data:
                path = mpltPath.Path(b)
                inside = path.contains_points([[xc,yc]])
                if inside[0]:
                    in_poly_mat[i][j] = 1
                    break
                    #            print(inside)
                    #            if isPointInPath(xc,yc,b):
                    #                in_poly_mat[i][j] = 1
                    #                break
    
    in_poly_mat = np.rot90(in_poly_mat)
    #plt.show()
    import pickle
    with open('nyc_grid.pkl', 'wb') as f:
        pickle.dump(in_poly_mat,f)
in_poly_mat = np.rot90(in_poly_mat,3)
potential_mat = np.zeros((len(x),len(y) ))
t_coord = []

for i in range(len(x)):
    if not i%100: print(i,'-',len(x))
    for j in range(len(y)):
        lib_in_range = 0
        for cl,l in enumerate(lib_coords):            
            #switch j and i coords because of rotation for plotting
            if in_poly_mat[i][j]:
                p_to_lib_dist = miles_dist([x[i],y[j]],l)
                if p_to_lib_dist < dist_cut - x_step:
                    lib_in_range += 1
#                    print(p_to_lib_dist//x_step,len(potential))
                    potential_mat[i][j] += potential[int(p_to_lib_dist//x_step)-1]#potential(p_to_lib_dist)
#        potential_mat[i][j] /= lib_in_range
#    t_coord.append(l)
#    plt.scatter(np.asarray(t_coord).T[0], np.asarray(t_coord).T[1])
#    h = plt.imshow(np.rot90(potential_mat),extent = [W - grid_size*0.5,E- grid_size*0.5,S - grid_size*0.5,N - grid_size*0.5])
#    plt.show()
#potential_mat = np.rot90(potential_mat)
#h = plt.imshow(in_poly_mat,extent = [W - grid_size*0.5,E - grid_size*0.5,S - grid_size*0.5,N - grid_size*0.5])


borders = [W - grid_size*0.5,E - grid_size*0.5,S - grid_size*0.5,N - grid_size*0.5]
pickle.dump([potential_mat,borders,lib_coords,potential,x_step],open('map_plot_data.pkl','wb'))

h = plt.imshow(np.exp(-np.rot90(potential_mat)),extent = [W - grid_size*0.5,E - grid_size*0.5,S - grid_size*0.5,N - grid_size*0.5])


plt.scatter(lib_coords.T[0], lib_coords.T[1])
plt.show()
