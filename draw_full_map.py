import sys,os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
#import streets_in_region
import pandas as pd
import copy
from matplotlib.collections import LineCollection
import subprocess
from matplotlib.patches import Circle

bounding_box = [-74.256771,-73.699229,40.497721,40.915140]

full = pd.read_csv('../predict_bike_lanes/Centerline.csv')
#print(list(full))


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
print(len(lib_coords))
exit(0)

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


def draw_roadmap(df,model_list):
    #colors based on bike lane type
    #class 1, class 2, class 3, link, class 1&2, class 2&3, stairs, class 1&3
    color_dict = {0:'black', 1:'red', 2:'green', 3:'blue', 4:'gray', 5:'pink', 6:'blue', 7:'black',8:'blue'}
    
    #colors based on regression models:
    segments = df['the_geom']
    #print(len(segments), len(df))
    seg_color = []
    seg_data = []
    for ci,i in enumerate(segments):
        if df['BIKE_LANE_log'][ci] != -1:
            seg_data_temp = i[18:-2].split(', ')
            #print(seg_data_temp)
            for j in range(0,len(seg_data_temp)-1):
                point_1 = seg_data_temp[j].split()
                point_2 = seg_data_temp[j+1].split()
                seg_data.append([(float(point_1[0]),float(point_1[1])), (float(point_2[0]),float(point_2[1]))])
    #            x_seg_data.append(None)
    #            y_seg_data.extend([float(point_1[1]), float(point_2[1])])
    #            y_seg_data.append(None)
                #print('ddd',seg_data[-1])
                log_val = df['BIKE_LANE_log'][ci]
    print(len(seg_color))
    fig = plt.figure()
    #fig.subplots_adjust(left=-0.15, right=1.15, bottom=-0.15, top=1.15)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=(bounding_box[0],bounding_box[1]), ylim=(bounding_box[2], bounding_box[3]))
    coll1 = LineCollection(seg_data, linewidths = 0.15, colors = 'grey')#seg_color)
    coll2 = LineCollection(boundary_data, linewidths = 0.5, color='black')
    ax.add_collection(coll1)
    ax.add_collection(coll2)
    for x,y in zip(lib_coords.T[0],lib_coords.T[1]):
        ax.add_artist(Circle(xy=(x,y), radius=0.0025*0.6,color='black',zorder=11))
        ax.add_artist(Circle(xy=(x,y), radius=0.0012*0.6,color='blue',zorder = 11))
    #rect = plt.Rectangle(bounding_box[::2],bounding_box[1] - bounding_box[0],bounding_box[3] - bounding_box[2],ec='none', lw=2, fc='none')
    #roadmap = np.asarray(seg_data)
    plt.axis('off')
    plt.savefig('library_streetmap.png',dpi=1200)
    plt.clf()
    print("green: correct (bike lane)\nyellow: correct (no bike lane)\nred: predict bike lane where there is no bike lane\nblue: predice no bike lane where there is a bike lane")



#'PHYSICALID', 'L_HIGH_HN', 'the_geom', 'L_LOW_HN', 'R_LOW_HN', 'R_HIGH_HN', 'L_ZIP', 'R_ZIP', 'L_BLKFC_ID', 'R_BLKFC_ID', 'ST_LABEL', 'STATUS', 'BIKE_LANE', 'BOROCODE', 'ST_WIDTH', 'CREATED', 'MODIFIED', 'TRAFDIR', 'RW_TYPE', 'FRM_LVL_CO', 'TO_LVL_CO', 'SNOW_PRI', 'PRE_MODIFI', 'PRE_DIRECT', 'PRE_TYPE', 'POST_TYPE', 'POST_DIREC', 'POST_MODIF', 'FULL_STREE', 'ST_NAME', 'BIKE_TRAFD', 'SHAPE_Leng'
#drop columns
full = full.drop(['PHYSICALID', 'L_HIGH_HN', 'L_LOW_HN', 'R_LOW_HN', 'R_HIGH_HN', 'L_BLKFC_ID','R_BLKFC_ID', 'ST_LABEL','CREATED','MODIFIED','PRE_MODIFI','PRE_DIRECT','PRE_TYPE','POST_DIREC','FULL_STREE','POST_MODIF','BIKE_TRAFD'], axis = 1)

#df.loc[(df['A'] == 'foo') & ...]

#print(full.head(10))

#1 Street 2 Highway 3 Bridge 4 Tunnel 5 Boardwalk 6 Path/Trail 7 StepStreet 8 Driveway 9 Ramp 10 Alley 11 Unknown 12 Non-Physical Street Segment 13 U Turn 14 Ferry Route
full = full.loc[(full['RW_TYPE'] != 8) & (full['RW_TYPE'] != 11) & (full['RW_TYPE'] != 12) & (full['RW_TYPE'] != 14)]

#dat.dropna(subset=['x'])
#V Non - DSNY C Critical H Haulster S Sector
full.dropna(subset = ['SNOW_PRI'],inplace=True)
unique_snow_pri = full['SNOW_PRI'].unique()
snow_targets = {i: ci for ci,i in enumerate(unique_snow_pri)}
print(snow_targets)
full['SNOW_PRI_num'] = full['SNOW_PRI'].replace(snow_targets)


#FT-With TF-Against TW-Two-Way NV -Non-Vehicular
full['TRAFDIR'].replace('FT','OO',inplace=True)
full['TRAFDIR'].replace('TF','OO',inplace=True)
full.dropna(subset = ['TRAFDIR'],inplace=True)
unique_traf_dir = full['TRAFDIR'].unique()
traf_targets = {i:ci for ci,i in enumerate(unique_traf_dir)}
print(traf_targets)
full['TRAFDIR_num'] = full['TRAFDIR'].replace(traf_targets)


full.dropna(subset = ['POST_TYPE'], inplace = True)
unique_post_type = full['POST_TYPE'].unique()
post_targets = {i:ci for ci,i in enumerate(unique_post_type)}
print(post_targets)
full['POST_TYPE_num'] = full['POST_TYPE'].replace(post_targets)

#    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

#1 Class I  2 Class II  3 Class III  4 Links  5 Class I, II  6 Class II, III  7 Stairs  8 Class I, III  9 Class II, I  10 Class III, I  11 Class III, II 

#bike_lane_types = full['BIKE_LANE'].unique()
#map_to_int = {name: n for n, name in enumerate(bike_lane_types)}
full['BIKE_LANE'].fillna(0,inplace=True)
full['BIKE_LANE'].replace(9,5,inplace=True)
full['BIKE_LANE'].replace(11,6,inplace=True)
full['BIKE_LANE'].replace(10,8,inplace=True)
full['BIKE_LANE'].replace(7,0,inplace=True)



#full.ix[full.BIKE_LANE == 0, 'BIKE_LANE_log'] = 0
full['BIKE_LANE_log'] = np.where(full.BIKE_LANE > 0, 1, np.where(full.BIKE_LANE == 0, 0, 0))

#print(full[['BIKE_LANE_log', 'BIKE_LANE']].head(20))

full.dropna(how='any',inplace=True)
full = full.reset_index()
#print(full.head(10))
#print(full.describe())

#train, test = train_test_split(full, test_size=0.5)
#
#print(len(train),len(test),len(full))
##features = ['STATUS', 'ST_WIDTH', 'TRAFDIR_num', 'RW_TYPE', 'FRM_LVL_CO', 'TO_LVL_CO', 'SNOW_PRI_num', 'POST_TYPE_num', 'SHAPE_Leng']
#features = ['ST_WIDTH', 'TRAFDIR_num', 'RW_TYPE', 'POST_TYPE_num', 'SHAPE_Leng', 'L_ZIP', 'R_ZIP']
#
#train_y = train['BIKE_LANE_log']
#train_X = train[features]
#test_y = test['BIKE_LANE_log']
#test_X = test[features]
#
#
#for i in range(5,100,10):
#    dt = DecisionTreeClassifier(min_samples_split=i, random_state=99)
#    dt.fit(train_X, train_y)
#    #print('dt done')
#    target_names = ['no bike lane', 'bike lane']
#    #print('dt, %d'%(i), classification_report(test_y, dt.predict(test_X), target_names=target_names))
#    
#    rf = RandomForestClassifier(n_estimators=5, max_depth=None,min_samples_split=i, random_state=0)
#    rf.fit(train_X, train_y)
#    print('rf, %d'%i, classification_report(test_y, rf.predict(test_X), target_names = target_names))
#    #print('rf done')
#
#
#
#
#rand = DummyClassifier()
#rand.fit(train_X, train_y)
#print('rand done')
#log = LogisticRegression()
#log.fit(train_X, train['BIKE_LANE_log'])
#print('log done')
##svm = svm.SVC()
##svm.fit(train_X, train['BIKE_LANE_log'])
#print('svm done')
#test = test.assign(pred_dt = dt.predict(test_X))
#test = test.assign(pred_rf = rf.predict(test_X))
#test = test.assign(pred_rand = rand.predict(test_X))
#test = test.assign(pred_log = log.predict(test_X))
##test = test.assign(pred_svm = svm.predict(test_X))
#model_list = ['pred_dt', 'pred_rf', 'pred_rand', 'pred_log']#,'pred_svm']
#matchrf = 0
#matchdt = 0
#matchrand = 0
#matchlog = 0
#matchsvm = 0
#total = 0
#for index, row in test.iterrows():
#    if row['BIKE_LANE_log'] == row['pred_rf']:
#        matchrf += 1
#    if row['BIKE_LANE_log'] == row['pred_dt']:
#        matchdt += 1
#    if row['BIKE_LANE_log'] == row['pred_rand']:
#        matchrand += 1
#    if row['BIKE_LANE_log'] == row['pred_log']:
#        matchlog += 1
#        #   if row['BIKE_LANE_log'] == row['pred_svm']:
#        #       matchsvm += 1
#    total += 1
#print('random forest: ', matchrf/total, '\ndecision tree: ', matchdt/total, '\nrandom: ', matchrand/total, '\nlogistic regression: ', matchlog/total, '\nsvm: ', matchsvm/total)
#test = test.reset_index()
model_list = []
draw_roadmap(full,model_list)

#with open("dt.dot", 'w') as f:
#    export_graphviz(dt, out_file=f,feature_names=features)
#command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
#try:
#    subprocess.check_call(command)
#except:
#    exit("Could not run dot, ie graphviz, to produce visualization")
#


