import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shift_cmap
import pickle
import glob
import matplotlib.path as mpltPath

pickle_files = glob.glob('*.pkl')
dist_cut = 10


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


N,S,W,E = 40.915568,40.495992,-74.257159,-73.699215
bounding_box = [W,E,S,N]#-74.256771,-73.699229,40.497721,40.915140]
#fig = plt.figure()
##fig.subplots_adjust(left=-0.15, right=1.15, bottom=-0.15, top=1.15)
#ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=(bounding_box[0],bounding_box[1]), ylim=(bounding_box[2], bounding_box[3]))
#coll = LineCollection(boundary_data, linewidths = 0.5, color='red')
#ax.add_collection(coll)
#
#rect = plt.Rectangle(bounding_box[::2],bounding_box[1] - bounding_box[0],bounding_box[3] - bounding_box[2],ec='none', lw=2, fc='none')
#roadmap = np.asarray(seg_data)
#plt.axis('off')
#plt.savefig('roadmap_%s.png'%(model),dpi=1200)



data = pickle.load(open('map_plot_data.pkl','rb'))
potential_mat = data[0]
borders = data[1]
lib_coords = data[2]
potential = data[3]
pot_step = data[4]
x_vals = np.linspace(0,len(potential)*pot_step,len(potential))
plt.plot(x_vals,potential)
plt.show()

#coll = data[3]

N,S,W,E = 40.915568,40.495992,-74.257159,-73.699215
bounding_box = [W,E,S,N]#-74.256771,-73.699229,40.497721,40.915140]
#new_lib_coords= []

prob = np.exp(-np.rot90(0.01*potential_mat))
cbounds = [0,1.6]#abs(np.nanmin(prob)),abs(np.nanmax(prob))]
orig_cmap = plt.cm.RdBu
shifted_cmap = shift_cmap.shiftedColorMap(orig_cmap, midpoint=(1-cbounds[0])/(cbounds[1]-cbounds[0]), name='shifted')
fig = plt.figure()
def plot_libs(potential_mat, new_lib_coords = []):

    #fig, ax = plt.subplots()
    #fig.subplots_adjust(left=-0.15, right=1.15, bottom=-0.15, top=1.15)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=(bounding_box[0],bounding_box[1]), ylim=(bounding_box[2], bounding_box[3]))
    coll = LineCollection(boundary_data, linewidths = 0.5, color='black')
    
    prob = np.exp(-0.02*np.rot90(potential_mat))
#    cbounds = [abs(np.nanmin(prob)),abs(np.nanmax(prob))]
#    print(cbounds)
    
    #shrunk_cmap = shiftedColorMap(orig_cmap, start=0.15, midpoint=0.75, stop=0.85, name='shrunk')
    
    norm = colors.Normalize(vmin=0.4,vmax=1.6)
    #

    im = ax.imshow(prob,extent = borders,norm=norm,cmap='RdBu')#shifted_cmap)
    for x,y in zip(lib_coords.T[0],lib_coords.T[1]):
        ax.add_artist(Circle(xy=(x,y), radius=0.0025,color='black'))
        ax.add_artist(Circle(xy=(x,y), radius=0.0012,color='yellow'))
    if len(new_lib_coords) > 0:
        for x,y in zip(new_lib_coords.T[0],new_lib_coords.T[1]):
            ax.add_artist(Circle(xy=(x,y), radius=0.0025,color='black'))
            ax.add_artist(Circle(xy=(x,y), radius=0.0012,color='red'))
        
    ax.add_collection(coll)#ax.scatter(lib_coords.T[0], lib_coords.T[1],color='yellow',s = 0.3)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(im, cax=cax)
    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.patch.set_visible(False)
    ax.axis('off')
    plt.tight_layout()

    

    
#plot_libs(potential_mat)
#plt.show()




grid_size = 0.0015

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

#t_coord = []
potential_mat = np.zeros((len(x),len(y) ))
potential_lib = np.zeros((len(x),len(y) ))

def make_potential_mat():
    for i in range(len(x)):
        if not i%100: print(i,'-',len(x))
        for j in range(len(y)):
            lib_in_range = 1
            for cl,l in enumerate(lib_coords):
                #switch j and i coords because of rotation for plotting
                if in_poly_mat[i][j]:
                    p_to_lib_dist = miles_dist([x[i],y[j]],l)
                    if p_to_lib_dist < dist_cut:
                        lib_in_range += 1
    #                    print(p_to_lib_dist//x_step,len(potential))
                        potential_mat[i][j] += potential[int(p_to_lib_dist//pot_step)]#potential(p_to_lib_dist)
            #potential_lib[i][j] += lib_in_range
    #    t_coord.append(l)
    #    plt.scatter(np.asarray(t_coord).T[0], np.asarray(t_coord).T[1])
    #    h = plt.imshow(np.rot90(potential_mat),extent = [W - grid_size*0.5,E- grid_size*0.5,S - grid_size*0.5,N - grid_size*0.5])
    #    plt.show()
    #potential_mat = np.rot90(potential_mat)
    #h = plt.imshow(in_poly_mat,extent = [W - grid_size*0.5,E - grid_size*0.5,S - grid_size*0.5,N - grid_size*0.5])
    
    
    borders = [W - grid_size*0.5,E - grid_size*0.5,S - grid_size*0.5,N - grid_size*0.5]
    #pickle.dump([potential_mat,borders,lib_coords,potential,x_step],open('map_plot_data.pkl','wb'))
    
    #h = plt.imshow(np.exp(-np.rot90(potential_mat)),extent = [W - grid_size*0.5,E - grid_size*0.5,S - grid_size*0.5,N - grid_size*0.5])
    
    
    #plt.scatter(lib_coords.T[0], lib_coords.T[1])
    #plt.show()



make_potential_mat()

#potential_mat = np.rot90(potential_mat)
prob = np.exp(-np.rot90(0.01*potential_mat))
#plt.clf()
new_lib_count = 0
plot_libs(potential_mat,[])
#    plt.show()
plt.savefig('heatmap_%03d.png'%(new_lib_count), dpi=200)


new_lib = []
x_step = x[1] - x[0]
y_step = y[1] - y[0]

potential[0] = potential[1]
while new_lib_count < 100:#np.nanmin(np.exp(-0.01*np.rot90(potential_mat))) < 0.8:
    #prob = np.exp(-0.01*np.rot90(potential_mat))
    #    print(np.amax(prob))
    print(potential[0],potential[1])
    max_loc = np.argwhere(potential_mat==np.amin(potential_mat))
    print(max_loc)
    new_lib.append([x[max_loc[0][0]]+0.5*x_step,y[max_loc[0][1]]+0.5*y_step])
    print(new_lib)
    #lib_coords = np.concatenate((lib_coords,np.asarray(new_lib)))
    #lib_in_range = 1
    #   for i in lib_coords[:-1]:
    #       lib_to_lib_dist = miles_dist(i,lib_coords[-1])
    #       if lib_to_lib_dist < dist_cut - pot_step:
    #           lib_in_range += 1
    for i in range(len(x)):
        if not i%100: print(i,'-',len(x))
        for j in range(len(y)):
            lib_in_range = 0
            #for cl,l in enumerate(lib_coords):
            #switch j and i coords because of rotation for plotting
            if in_poly_mat[i][j]:
                p_to_lib_dist = miles_dist([x[i],y[j]],new_lib[new_lib_count])
                if p_to_lib_dist < dist_cut:
                    lib_in_range += 1
                    #                    print(p_to_lib_dist//pot_step,len(potential))
                    potential_mat[i][j] += potential[int(p_to_lib_dist//pot_step)]
    new_lib_count += 1
    new_lib_coords = np.asarray(new_lib)
    plot_libs(potential_mat,new_lib_coords)
#    plt.show()
    plt.savefig('heatmap_%03d.png'%(new_lib_count), dpi=200)
    plt.clf()
    #plt.show()
