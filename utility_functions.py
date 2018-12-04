import os
import numpy as np
import pandas as pd
import time
import pickle
from simpledbf import Dbf5
from matplotlib import pyplot as plt
import networkx as nx
import math
import ogr
import osr
from pyproj import Proj, transform
import datetime
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn import preprocessing
from collections import defaultdict
from pyproj import Proj, transform

#read data from the road network shapefiles
def CreateNetworkFromSHP(shp_filename, verbose=False):
    start = time.time()
    G = nx.read_shp(shp_filename)
    G = G.to_undirected()

    if verbose:
        end = time.time()
        print end-start, 'seconds to create road network graphs'

    return G

    
#read in the shapefiles and clean data records that are not to be used in the model
#shapefile preprocessing to replace 0's by None
def shp_preproc(filepath, threshold_yr, output_nam):
    G = CreateNetworkFromSHP(filepath, verbose=True)
    
    #set invalid AADT as Null (e.g. 0 and records before the desired time, such <2013 for 2014; and <2014 for 2015 - 2017)
    start = time.time()
    zeros = 0
    bad_years = 0
    for edge in G.edges_iter(data=True):
        data = edge[2]
        if data['AADT'] == 0.0:
            del data['AADT']
            zeros += 1
        elif int(data['AADT_YR'] ) < threshold_yr:
            #int(data['AADT_YR']) < 2013 for year 2014
            del data['AADT']
            bad_years += 1
	
	if not os.path.exists(os.getcwd() + '/output'):
		os.makedirs(os.getcwd() + '/output')

    output = os.getcwd() + '/output/' + output_nam
    pickle.dump(G, open(output, 'wb'))
    print time.time() - start, "to removed", zeros, "0's and", bad_years, "records from earlier years, and pickle the processed networkx graph."
    
#read in data from dbf files and run the first round EDA     
def EDA_dbf(filepath_dbf, X_exc1):
    dbf = Dbf5(filepath_dbf)
    df = dbf.to_dataframe()
    
    X_col = set(df.columns)
    X_col = X_col - X_exc1
    
    nul = []
    one_val = []
    for col in X_col:
        if sum(pd.notnull(df[col])) < 0.8*df.shape[0]:
            nul.append(col)
            continue
        
        val = len(np.unique(np.array(df[col])))
        if val < 2:
            one_val.append(col)
        
    det = nul + one_val
    return set(det), set(df.columns)    

#decode and visualize the results generated after one-hot-encoding
def OHE_decoder(X,Y,X_cat,X_num,ohe,array,aggregate=None,visualize=False,output=False):
    decoded={}
    array_num = array[:len(X_num)]
    array_cat = array[len(X_num):]
    Feature_Indices = ohe.feature_indices_
    for i in xrange(len(X_num)):
        decoded[X_num[i]] = array_num[i]
    for i in xrange(len(X_cat)):
        start = Feature_Indices[i]
        end = Feature_Indices[i+1]
        decoded[X_cat[i]] = array_cat[start:end]
        
        if aggregate == 'sum':
            decoded[X_cat[i]] = np.sum(decoded[X_cat[i]])
        elif aggregate == 'average':
            decoded[X_cat[i]] = np.mean(decoded[X_cat[i]])
        elif aggregate == 'max':
            decoded[X_cat[i]] = np.max(decoded[X_cat[i]])
                
    if visualize == True:
        if aggregate == None:
            print "Please specify an aggregation method first."
        else:
            x = np.arange(len(decoded))
            mi = np.array([v for k,v in decoded.items()])
            plt.bar(x, mi)
            plt.xticks(x, [k for k,v in decoded.items()], rotation='vertical')
            plt.show()
    
	if output == True:
		var = [k for k,v in decoded.items()]
		val = [v for k,v in decoded.items()]
		result = pd.DataFrame({'Variable':var, 'Values':val})
		return result
        
def feature_selection_helper(X, Y, X_cat, X_num, ohe, tool='MIR', aggregate=None, visualize=False):
    if tool == 'MIR':
        Mutual_Info = mutual_info_regression(X, Y)
        OHE_decoder(X, Y, X_cat, X_num, ohe, Mutual_Info, aggregate, visualize)
    if tool == 'MIC':
        Mutual_Info = mutual_info_classif(X, Y)
        OHE_decoder(X, Y, X_cat, X_num, ohe, Mutual_Info, aggregate, visualize)
        
#a nested cross validation tool to find the best hyperparameters for the model        
def optimizer (model, grid, NUM_TRIALS, exposureX, exposureY):
    cv_scores_inner = np.zeros(NUM_TRIALS)
    cv_scores_outer = np.zeros(NUM_TRIALS)
    cv_best_n = []
    
    for i in xrange(NUM_TRIALS):
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=3, shuffle=True, random_state=i)
        
        rgr = GridSearchCV(estimator=model, param_grid=grid, cv=inner_cv)
        rgr.fit(exposureX, exposureY)
        cv_scores_inner[i] = rgr.best_score_
        cv_best_n.append(rgr.best_params_)
        
        outer_score = cross_val_score(rgr, X=exposureX, y=exposureY, cv=outer_cv)
        cv_scores_outer[i] = outer_score.mean()
        
    return cv_scores_inner, cv_scores_outer, cv_best_n

#attach lat-lon events to the roadnetwork
def attach_info(p1x, p1y, p2x, p2y, G, x, y, value, attributes=None, duplicates='overwrite', skipzeros=True):
    # Longitude is x value
    # Latitude is y value

    if p1x is None:
        p1x = []
        p1y = []
        p2x = []
        p2y = []
        for edge in G.edges_iter():
            p1x.append(edge[0][0])
            p1y.append(edge[0][1])
            p2x.append(edge[1][0])
            p2y.append(edge[1][1])

        p1x = np.array(p1x)
        p1y = np.array(p1y)
        p2x = np.array(p2x)
        p2y = np.array(p2y)

    p1x = p1x
    p1y = p1y
    p2x = p2x
    p2y = p2y
    # Calc distance between point and the closest point on each edge
    dist = abs((p2y - p1y) * x - (p2x - p1x) * y + p2x * p1y - p2y * p1x) / np.sqrt(
        (p2y - p1y) ** 2 + (p2x - p1x) ** 2)

    # Check if the point is within the x_range or y_range of an edge
    p1x_greater_than = abs(p1x) <= abs(x)  # p1x <= x
    p2x_less_than = abs(x) <= abs(p2x)  # x <= p2x
    p1y_greater_than = abs(p1y) <= abs(y)  # p1y <= y
    p2y_less_than = abs(y) <= abs(p2y)  # y <= p2y

    # Any edge that the point is not in range of, is disqualified
    x_inbetween = np.logical_and(p1x_greater_than, p2x_less_than)
    y_inbetween = np.logical_and(p1y_greater_than, p2y_less_than)
    inbetween = np.logical_or(x_inbetween, y_inbetween)

    # All disqualified edges are given value 'nan'
    dist = dist * inbetween
    dist[np.where(dist == 0)] = np.nan

    # find the index of the closest edge
    closest_index = np.nanargmin(dist)
    closest_edge = ((p1x[closest_index], p1y[closest_index]), (p2x[closest_index], p2y[closest_index]))
    for i,attribute in enumerate(attributes):
        if value[i] == 0 and skipzeros is True:
            pass
        else:
            if attribute in G[closest_edge[0]][closest_edge[1]]:
                if duplicates == 'overwrite':
                    nx.set_edge_attributes(G, attribute, {closest_edge: [value[i]]})
                elif duplicates == 'append':
                    G[closest_edge[0]][closest_edge[1]][attribute].append(value[i])
            else:
                nx.set_edge_attributes(G, attribute, {closest_edge: [value[i]]})

                
#prepare data to be fed to the ML estimators (i.e. label encoding, one hot encoding, vectorization)
def get_data_for_training_preprocessed(G, X_columns=None, Y_column=None, X_cat=None, X_int=None, X_float=None, multiple_x='newest', multiple_y='newest', verbose=False):
        samples = []
        X_c = X_columns
        for edge in G.edges_iter(data=True):
            data = edge[2]
            if Y_column in data:
                midpoint = ((edge[0][0] + edge[1][0]) / 2, (edge[0][1] + edge[1][1]) / 2)

                sample = [midpoint[0], midpoint[1]]

                for column in X_columns:
                    try:
                        if isinstance(data[column], list):
                            if multiple_x == 'newest':
                                sample.append(data[column][-1])
                            if multiple_x == 'sum':
                                sample.append(sum(data[column]))
                            if multiple_x == 'average':
                                sample.append(sum(data[column]) / len(data[column]))
                        else:
                            sample.append(data[column])
                    except:
                        sample.append(None)

                if isinstance(data[Y_column], list):
                    if multiple_y == 'newest':
                        sample.append(data[Y_column][-1])
                    if multiple_y == 'sum':
                        sample.append(sum(data[Y_column]))
                    if multiple_y == 'average':
                        sample.append(sum(data[Y_column])/len(data[Y_column]))
                else:
                    sample.append(data[Y_column])


                samples.append(sample)


        if verbose: print 'done creating model training data with ' + str(len(samples)) + " samples"

        data_df = pd.DataFrame(samples)

        col = ['X', 'Y']
        col = col + X_c
        col.append('attribute')
        data_df.columns = col

        #data_df.to_csv('C:/Users/husiy/PyProgram/OPEN DATA NATION/Chicago_Test/test710.csv', index=False)

        cl = data_df.columns.get_values()
        # print cl
        det = []
        # dl=xrange(len(cl))
        for c in cl:
            # print sum(pd.notnull(data_df.iloc[:,c]))
            if sum(pd.notnull(data_df[c])) <= 0.8 * data_df.shape[0]:
                det.append(c)
                # dl.append(c-2)
                # for il in sorted(dl, reverse=True):
                # del X_c[il]
        # print det
        data_df = data_df.drop(det, 1)
        data_df = data_df.dropna()
        for dc in det:
            X_c.remove(dc)
            # print X_c
            # print 'dropna', data_df.shape

            # print data_df.head()
            # print data_df.dtypes

        # print data_df.head()


        if X_cat != None:
            data_df[X_cat] = data_df[X_cat].astype(str)
        if X_int != None:
            data_df[X_int] = data_df[X_int].astype('int64')
        if X_float != None:
            data_df[X_float] = data_df[X_float].astype('float64')
        print data_df.dtypes

        data_df_catagorical = data_df.select_dtypes(exclude=['float64', 'int64'])
        data_df_numerical = data_df.select_dtypes(include=['float64', 'int64'])
        # TODO one hot encode the catagoricals and start training a model
        if len(data_df_catagorical.columns) != 0:

            # TODO one hot encode the catagoricals and start training a model
            ohe = preprocessing.OneHotEncoder(sparse=False)

            d = defaultdict(preprocessing.LabelEncoder)
            data_df_labelenc = data_df_catagorical.apply(lambda x: d[x.name].fit_transform(x))

            # print data_df_catagorical
            # print data_df_labelenc.values

            x_ohe = ohe.fit_transform(data_df_labelenc.values)

            x_preprocessed = np.concatenate((data_df_numerical.values[:, 0:data_df_numerical.shape[1] - 1], x_ohe), axis=1)
        else:
            x_preprocessed = data_df_numerical.values[:, 0:data_df_numerical.shape[1] - 1]
        
        sscaler = preprocessing.MinMaxScaler()
        sscaler.fit(x_preprocessed[:, 0:data_df_numerical.shape[1]-1])
        x_preprocessed[:, 0:data_df_numerical.shape[1]-1] = sscaler.transform(x_preprocessed[:, 0:data_df_numerical.shape[1]-1])
        y_preprocessed = data_df_numerical.values[:, -1]

        or_x_preprocessed = sscaler.inverse_transform(x_preprocessed[:, 0:data_df_numerical.shape[1] - 1])
        or_x_preprocessed = pd.DataFrame(or_x_preprocessed)
        n_c=data_df_numerical.columns.values.tolist()
        n_c.remove('attribute')
        or_x_preprocessed.columns = n_c

        # Saving transformations for later use
        try:
            OHE = ohe
            LabelEncoder = d
        except:
            pass

        return x_preprocessed,y_preprocessed, ohe, d, sscaler
    
    
    
def fill_network_data_from_model(G, ohe, d, sscaler, X_columns, categorical, floating, integer, attribute, trained_model):
    ohe = ohe
    d = d
    sscaler = sscaler
    er=0
    deg=[]
    for edge in G.edges_iter(data=True):

        #try:

        data = edge[2]
        samples = []
        if attribute not in data:
            midpoint = ((edge[0][0] + edge[1][0]) / 2, (edge[0][1] + edge[1][1]) / 2)
            sample = [midpoint[0], midpoint[1]]

            for att in X_columns:
                sample.append(data[att])

            samples.append(sample)


            data_df = pd.DataFrame(samples)
            col=['X','Y']
            col=col+X_columns
            data_df.columns=col

            data_df[categorical] = data_df[categorical].astype(str)
            data_df[floating] = data_df[floating].astype('float64')
            data_df[integer] = data_df[integer].astype('int64')

            data_df_catagorical = data_df.select_dtypes(exclude=['float64', 'int64'])
            data_df_numerical = data_df.select_dtypes(include=['float64', 'int64'])


            if np.any(np.equal(data_df.values[0], None)) != True:
                data_df_labelenc = data_df_catagorical.apply(lambda x: d[x.name].transform(x))

                new_X = ohe.transform(data_df_labelenc.values)

                new_X = np.concatenate((data_df_numerical.values[:, 0:data_df_numerical.shape[1]], new_X), axis=1)
                new_X[:, 0:data_df_numerical.shape[1]] = sscaler.transform(new_X[:, 0:data_df_numerical.shape[1]])
                new_Y = trained_model.predict(new_X)
                nx.set_edge_attributes(G, attribute, {(edge[0], edge[1]): [new_Y[0]]})
#         except:
#             er+=1
#             print er,' something wrong for edge data',edge[2]
#             deg.append(edge)
#             pass

def predict_risk(G, ohe, d, sscaler, X_columns, categorical, floating, integer, attribute, trained_model):
    ohe = ohe
    d = d
    sscaler = sscaler
    er=0
    for edge in G.edges_iter(data=True):
        try:
            data = edge[2]
            samples = []
            if attribute not in data:
                midpoint = ((edge[0][0] + edge[1][0]) / 2, (edge[0][1] + edge[1][1]) / 2)
                sample = [midpoint[0], midpoint[1]]

                for att in X_columns:
                    sample.append(data[att])

                samples.append(sample)


                data_df = pd.DataFrame(samples)
                col=['X','Y']
                col=col+X_columns
                data_df.columns=col

                data_df[categorical] = data_df[categorical].astype(str)
                data_df[floating] = data_df[floating].astype('float64')
                data_df[integer] = data_df[integer].astype('int64')

                data_df_catagorical = data_df.select_dtypes(exclude=['float64', 'int64'])
                data_df_numerical = data_df.select_dtypes(include=['float64', 'int64'])


                if np.any(np.equal(data_df.values[0], None)) != True:
                    data_df_labelenc = data_df_catagorical.apply(lambda x: d[x.name].transform(x))

                    new_X = ohe.transform(data_df_labelenc.values)

                    new_X = np.concatenate((data_df_numerical.values[:, 0:data_df_numerical.shape[1]], new_X), axis=1)
                    new_X[:, 0:data_df_numerical.shape[1]] = sscaler.transform(new_X[:, 0:data_df_numerical.shape[1]])
                    new_Y = trained_model.predict_proba(new_X)
                    nx.set_edge_attributes(G, attribute, {(edge[0], edge[1]): new_Y[0][1]})
        except:
           er+=1
           pass
    print "Done predicting risks!"
				
#functions that ingest postive and negative samples and produce a combined training dataset
def get_training_data(X_columns, categorical, floating, integer, pos_data, neg_data, sample_size=0.1, seed=9527):
    #sythesize a training dataset from positive and negative data
    #positive dataset is smaller
    pos_clear = pos_data[pos_data['WeatherCond'] == 'Clear']
    pos_other = pos_data[pos_data['WeatherCond'] != 'Clear']
    #16000 was chosen temporarily because about 16000 crashes are associated with rainy weather, 
    #and this is the about the same order of magnitutde compared to other weather conditions
    pos_clear_sub = pos_clear.sample(16000, random_state=seed)
    pos_data = pd.concat([pos_clear_sub, pos_other])
    
    sample_size = int(min(neg_data.shape[0], pos_data.shape[0]) * sample_size)
    pos_data['Crash'] = [1 for i in xrange(len(pos_data))]
    neg_data['Crash'] = [0 for i in xrange(len(neg_data))]
    
    columns = X_columns + ['Crash']
    data_df_pos = pos_data.sample(sample_size, random_state=seed)
    data_df_neg = neg_data.sample(sample_size, random_state=seed)
    data_df = pd.concat([data_df_pos[columns], data_df_neg[columns]])
    
    data_df[categorical] = data_df[categorical].astype(str)
    data_df[floating] = data_df[floating].astype('float64')
    data_df[integer] = data_df[integer].astype('int64')
    
    data_df_catagorical = data_df.select_dtypes(exclude=['float64', 'int64'])
    data_df_numerical = data_df.select_dtypes(include=['float64', 'int64'])
    # TODO one hot encode the catagoricals and start training a model
    ohe = preprocessing.OneHotEncoder(sparse=False)

    d = defaultdict(preprocessing.LabelEncoder)
    data_df_labelenc = data_df_catagorical.apply(lambda x: d[x.name].fit_transform(x))

    # print data_df_catagorical
    # print data_df_labelenc.values

    x_ohe = ohe.fit_transform(data_df_labelenc.values)

    x_preprocessed = np.concatenate((data_df_numerical.values[:, 0:data_df_numerical.shape[1]-1], x_ohe), axis=1)
    # TODO don't scale before spliting into training and test set. Added parameter that changes if the return is (x,y) or (x_train,ytrain, xtest, y test)
    # TODO change this to MaxMin scalar to avoid distorting coordinate data with a normal distribution
    # sscaler = preprocessing.StandardScaler()
    sscaler = preprocessing.MinMaxScaler()
    sscaler.fit(x_preprocessed[:, 0:data_df_numerical.shape[1]-1])
    x_preprocessed[:, 0:data_df_numerical.shape[1]-1] = sscaler.transform(x_preprocessed[:, 0:data_df_numerical.shape[1]-1])
    y_preprocessed = data_df_numerical.values[:, -1]

    #or_x_preprocessed = sscaler.inverse_transform(x_preprocessed[:, 0:data_df_numerical.shape[1] - 1])
    #or_x_preprocessed = pd.DataFrame(or_x_preprocessed)
    #n_c=data_df_numerical.columns.values.tolist()
    #or_x_preprocessed.columns = n_c
    
    return x_preprocessed,y_preprocessed, ohe, d, sscaler
	
#There are many records with 0 as latitude and longitude but many of those records have valid state plane coordinate. Here define a conversion function to impute as many the missing lat, lon 
#State Plane Coordinate used in csv: Illinois West 1202. Unit: US feet
def coord_converter(x1, y1, inCoord='esri:102672',outCoord='epsg:4326'):
    inProj = Proj(init=inCoord, preserve_units=True)
    outProj = Proj(init=outCoord)
    lon, lat = transform(inProj,outProj,x1,y1)
    
    return (lat, lon)

def crash_preprocessing(crashID):
    #crashID = pd.read_csv(filename, usecols=[1,44,45,46,47])
    originalLen = len(crashID)
    
    counter = 0
    for i in xrange(len(crashID)):
        if crashID.loc[i,'TSCrashLatitude'] == 0.0 and crashID.loc[i,'TSCrashCoordinateX'] != 0.0:
            crashID.loc[i,'TSCrashLatitude'] = coord_converter(crashID.loc[i,'TSCrashCoordinateX'],crashID.loc[i,'TSCrashCoordinateY'])[0]
            crashID.loc[i,'TSCrashLongitude'] = coord_converter(crashID.loc[i,'TSCrashCoordinateX'],crashID.loc[i,'TSCrashCoordinateY'])[1]
            counter += 1
    print "Imputed", counter, "lat, lon pairs"
    
    crashID = crashID[crashID['TSCrashLatitude']!= 0.0]
    print originalLen - len(crashID), "records were removed."
    return crashID

#extract the years from the 311 reports
def extract_year(df):
    strpyr = []
    format_str = '%m/%d/%Y'
    for i in xrange(df.shape[0]):
        try:
            date_obj = datetime.datetime.strptime(df['CREATION DATE'][i], format_str)
        except:
            date_obj = datetime.datetime.strptime(df['Creation Date'][i], format_str)
        strpyr.append(date_obj.year)
    df['CREATION YEAR'] = strpyr
	
	
#attach the crash record information to the road network
def attach_crash(G, dataset, col_crash, desig_name_crash):
    #first extract the lat-lon's of the two ends of the road segments
    start = time.time()
    p1x = []
    p1y = []
    p2x = [] 
    p2y = []
    for edge in G.edges_iter():
        p1x.append(edge[0][0])
        p1y.append(edge[0][1])
        p2x.append(edge[1][0])
        p2y.append(edge[1][1])

    p1x = np.array(p1x)
    p1y = np.array(p1y)
    p2x = np.array(p2x)
    p2y = np.array(p2y)
    print time.time()-start, 'seconds to extract the lat/lons of all the road segments.'
    
    start = time.time()
    l = 0
    for p in xrange(dataset.shape[0]):
        try:
            attach_info(p1x, p1y, p2x, p2y, G, dataset.iloc[p]['Longitude'], dataset.iloc[p]['Latitude'], [dataset.iloc[p][col] for col in col_crash], desig_name_crash, duplicates='append')
            l += 1
        except:
           pass
    print time.time() - start, 'seconds to attach', l, 'records.'
	
#attach other data to the road network
def attach_other(G, datasets,data,desig_name):  
    #first extract the lat-lon's of the two ends of the road segments
    start = time.time()
    p1x = []
    p1y = []
    p2x = [] 
    p2y = []
    for edge in G.edges_iter():
        p1x.append(edge[0][0])
        p1y.append(edge[0][1])
        p2x.append(edge[1][0])
        p2y.append(edge[1][1])

    p1x = np.array(p1x)
    p1y = np.array(p1y)
    p2x = np.array(p2x)
    p2y = np.array(p2y)
    print time.time()-start, 'seconds to extract the lat/lons of all the road segments.'

    i = 0
    for dataset in datasets:
        start=time.time()
        l=0
        for p in xrange(dataset.shape[0]):
            try:
                attach_info(p1x, p1y, p2x, p2y, G,dataset.iloc[p]['Longitude'],dataset.iloc[p]['Latitude'],[dataset[data[i]][p]],[desig_name[i]],duplicates='append')
                l+=1
            except:
                pass
        print time.time() - start, 'seconds to attach',l,'records'
        i += 1
		
#create fields that counts the number of incidents
def convert_to_count(G, filename):
    c = 0
    for edge in G.edges_iter(data=True):
        data = edge[2]
        try:
            data['Potholes'] = len(data['Potholes'])
        except:
            data['Potholes'] = 0
        try:
            data['StreetLightAll'] = len(data['StreetLightAll'])
        except:
            data['StreetLightAll'] = 0
        try:
            data['StreetLightOne'] = len(data['StreetLightOne'])
        except:
            data['StreetLightOne'] = 0
        try:
            data['TreeDeb'] = len(data['TreeDeb'])
        except:
            data['TreeDeb'] = 0
        try:
            data['TreeTrim'] = len(data['TreeTrim'])
        except:
            data['TreeTrim'] = 0
        try:
            data['CrashCnt'] = len(data['CrashID'])
        except:
            data['CrashCnt'] = 0
        try: 
            data['FatalCnt'] = len(data['TotalFatals'])
        except:
            data['FatalCnt'] = 0
        try:
            data['InjCnt'] = len(data['TotalInjured'])
        except:
            data['InjCnt'] = 0
            c += 1
    print c
    pickle.dump(G, open("./output/"+filename, 'wb'))
	

#Will need to create a list of road segments with no crashes in both 2015 and 2016
def get_crash_number_distribution(roadnetworks, percentile, visualize=False):
    #create a list of latitude and longitudes that are common to all the shapefiles
    common_seg = set([edge[2]['SEG_ID'] for edge in roadnetworks[0].edges_iter(data=True)])
    for roadnetwork in roadnetworks[1:]:
        common_seg.intersection_update([edge[2]['SEG_ID'] for edge in roadnetwork.edges_iter(data=True)])
    
    #empty list to store crash counts
    crash_cnt = [0 for i in xrange(len(common_seg))]
    common_seg = list(common_seg)
    
    for roadnetwork in roadnetworks:
        for edge in roadnetwork.edges_iter(data=True):
            data = edge[2]
            if data['SEG_ID'] in common_seg:
                ix = common_seg.index(data['SEG_ID'])
                try:
                    crash_cnt[ix] = crash_cnt[ix] + len(data['CrashID'])
                except:
                    pass
                
    crash_cnt = np.array(crash_cnt)
    print "Total number of common road segments:", len(common_seg)
    print percentile, "th percentile is", np.percentile(crash_cnt, percentile)
    print "The maximum number of crashes on a road segment for the time of interest is:", np.max(crash_cnt)
    print len(crash_cnt)-np.count_nonzero(crash_cnt), "road segments had no crashes for the time period of interest."
    
    if visualize == True:
        n, bins, patches = plt.hist(crash_cnt, 100, density=True, facecolor='b', alpha=0.75)
        
        plt.xlabel('Number of crashes')
        plt.ylabel('Proportion of road segments')
        #for better visualization, the x axis is set to only show the bulk of the data (excluding outliers)
        plt.axis([0, 150, 0, 0.15])
        plt.grid(True)
        plt.show()
    
    crash_counts = pd.DataFrame({'SEG_ID':common_seg, 'Crash counts':crash_cnt})
    return crash_counts	
	
#positive sampling
def get_crash_for_training(X_columns, roadnetworks, crashreports):
    crashes = []
    p = 0
    for roadnetwork in roadnetworks:
        crashID = []
        sample = {k:[] for k in X_columns}
        Xs = []
        Ys = []
        
        for edge in roadnetwork.edges_iter(data=True):
            X = (edge[0][0] + edge[1][0])/2
            Y = (edge[0][1] + edge[1][1])/2
            
            try:
                for i in edge[2]['CrashID']:
                    crashID.append(i)
                    for col in X_columns:
                        if isinstance(edge[2][col], list):
                            sample[col].append(edge[2][col][-1])
                        else:
                            sample[col].append(edge[2][col])
                    Xs.append(X)
                    Ys.append(Y)
            except:
                pass
        
        samples = pd.DataFrame(sample)
        samples['X'] = Xs
        samples['Y'] = Ys
        samples['CrashID'] = crashID
        
        crash = samples.merge(crashreports[p], on=['CrashID'], how='left')
        crashes.append(crash)
        
        p += 1
        
    AllCrash = pd.concat(crashes,axis=0, ignore_index=True)
    print "Exporting to csv..."
    AllCrash.to_csv("AllCrashes.csv")
    print 'Done!'

#negative sampling
def get_nocrash_for_training(X_columns, roadnetworks, crash_counts, threshold=0):
    nocrash = crash_counts[crash_counts['Crash counts'] == threshold]['SEG_ID']
    
    neg_samples = []
    
    for roadnetwork in roadnetworks:
        road_seg = []
        sample = {k:[] for k in X_columns}
        Xs = []
        Ys = []
        
        for edge in roadnetwork.edges_iter(data=True):
            X = (edge[0][0] + edge[1][0])/2
            Y = (edge[0][1] + edge[1][1])/2
            data= edge[2]
            
            if data['SEG_ID'] in np.array(nocrash):
                road_seg.append(data['SEG_ID'])
                for col in X_columns:
                    if isinstance(data[col], list):
                        sample[col].append(data[col][-1])
                    else:
                        sample[col].append(data[col])
                Xs.append(X)
                Ys.append(Y)
        
        samples = pd.DataFrame(sample)
        samples['X'] = Xs
        samples['Y'] = Ys
        samples['SEG_ID'] = road_seg
        
        neg_samples.append(samples)
	
	NoCrash = pd.concat(neg_samples,axis=0, ignore_index=True)
    original_len = NoCrash.shape[0]
    
    #remove duplicated entries
    #TODO think through the methodology of keeping these entries, because road segments with no crash for the time period of interest
    #may have varied features (e.g. road condition, number of 311 reports). For now, only road segments with identical feature info were merged.
    #In other words, the same road segment with different variable values across say, two years are considered as two distinct samples.
    
    NoCrash.drop_duplicates(inplace=True)
    print original_len - NoCrash.shape[0],"identical recoreds dropped."
    
    #create negative sample sets by expand each road segment into 24 hours*7days/week*12month/year 
    print "Cloning info for road segments..."
    start = time.time()
    no_crash_all = pd.DataFrame(np.repeat(NoCrash.values, 24*7, axis=0))
    no_crash_all.columns = NoCrash.columns
    print "Elapsed time:", time.time()-start
    
    print "Creating series for hours of the day, day of week and month of the year..."
    hours = [i for i in xrange(24)]*(7*NoCrash.shape[0])
    
    DayOfWeek = []
    for j in xrange(1,8):
        DayOfWeek = DayOfWeek + [j for i in xrange(24)]
    DayOfWeek = DayOfWeek*NoCrash.shape[0]
    
    #CrashMonth = []
    #for j in xrange(1,13):
    #CrashMonth  = CrashMonth + [j for i in xrange(24*7)]
    #CrashMonth = CrashMonth*NoCrash.shape[0]
    
    no_crash_all['CrashHour'] = hours
    no_crash_all['DayOfWeekCode'] = DayOfWeek
    #no_crash_all['CrashMonth'] = CrashMonth
    
    print "Exporting to csv..."
    no_crash_all.to_csv("NoCrashes.csv")
    print "Done!"
		
#attach the crash record information to the road network
def attach_crash(G, dataset, col_crash, desig_name_crash):
    #first extract the lat-lon's of the two ends of the road segments
    start = time.time()
    p1x = []
    p1y = []
    p2x = [] 
    p2y = []
    for edge in G.edges_iter():
        p1x.append(edge[0][0])
        p1y.append(edge[0][1])
        p2x.append(edge[1][0])
        p2y.append(edge[1][1])

    p1x = np.array(p1x)
    p1y = np.array(p1y)
    p2x = np.array(p2x)
    p2y = np.array(p2y)
    print time.time()-start, 'seconds to extract the lat/lons of all the road segments.'
    
    start = time.time()
    l = 0
    for p in xrange(dataset.shape[0]):
        try:
            attach_info(p1x, p1y, p2x, p2y, G, dataset.iloc[p]['Longitude'], dataset.iloc[p]['Latitude'], [dataset.iloc[p][col] for col in col_crash], desig_name_crash, duplicates='append')
            l += 1
        except:
           pass
    print time.time() - start, 'seconds to attach', l, 'records.'
	
#attach other data to the road network
def attach_other(G, datasets,data,desig_name):  
    #first extract the lat-lon's of the two ends of the road segments
    start = time.time()
    p1x = []
    p1y = []
    p2x = [] 
    p2y = []
    for edge in G.edges_iter():
        p1x.append(edge[0][0])
        p1y.append(edge[0][1])
        p2x.append(edge[1][0])
        p2y.append(edge[1][1])

    p1x = np.array(p1x)
    p1y = np.array(p1y)
    p2x = np.array(p2x)
    p2y = np.array(p2y)
    print time.time()-start, 'seconds to extract the lat/lons of all the road segments.'

    i = 0
    for dataset in datasets:
        start=time.time()
        l=0
        for p in xrange(dataset.shape[0]):
            try:
                attach_info(p1x, p1y, p2x, p2y, G,dataset.iloc[p]['Longitude'],dataset.iloc[p]['Latitude'],[dataset[data[i]][p]],[desig_name[i]],duplicates='append')
                l+=1
            except:
                pass
        print time.time() - start, 'seconds to attach',l,'records'
        i += 1
		
#create a unique ID number for each road segment
def seg_encoder(G):
    for edge in G.edges_iter(data=True):
        data = edge[2]
        p1 = data['INVENTORY']
        p2 = str(int(data['BEG_STA']*100))
        p3 = str(int(data['END_STA']*100))
        
        if len(p2) == 1:
            p2 = '000' + p2
        elif len(p2) == 2:
            p2 = '00' + p2
        elif len(p2) == 3:
            p2 = '0' + p2
            
        if len(p3) == 1:
            p3 = '000' + p3
        elif len(p3) == 2:
            p3 = '00' + p3
        elif len(p3) == 3:
            p3 = '0' + p3
        
        data['SEG_ID'] = p1 + p2 + p3
		
#fill the road network with user-defined hypothetical variable values. Right now it can only take CrashHour
def define_day_hour(G, hour, dayofweek):
    #day of week dictionary
    DOW = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday': 5, 'Saturday':6, 'Sunday':7}
    
    for edge in G.edges_iter(data=True):
        edge[2]['CrashHour'] = hour
        edge[2]['DayOfWeek'] = str(dayofweek)
        edge[2]['hr_sin'] = np.sin(hour*2*np.pi/24)
        edge[2]['hr_cos'] = np.cos(hour*2*np.pi/24)
        edge[2]['day_sin'] = np.sin((DOW[dayofweek]-1)*2*np.pi/7)
        edge[2]['day_cos'] = np.cos((DOW[dayofweek]-1)*2*np.pi/7)
    print "Done inserting customized variables!"
	
	
def predict_crash_risk(G, hour, dayofweek, predictors, pred_cat, pred_flo, pred_int, trained_model, ohe, labEncoder, sscaler, csv_file):#, json_file
    start = time.time()
    define_day_hour(G, hour, dayofweek)
    predict_risk(G, ohe, labEncoder, sscaler, predictors, pred_cat, pred_flo, pred_int, 'Risk', trained_model)
    save_to_csv(G, predictors, pred_cat, pred_flo, pred_int, csv_file)
    #save_to_geojson(csv_file, json_file)
    print time.time() - start, "seconds to execute the prediction and save the results."
	
	
def save_to_geojson(csv_name, json_name):
    features = []
    with open(csv_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader,None)#skip the header line
        #the order of this list should match the order of the variables in the csv file
        for Xcoor1, Ycoor1, Xcoor2, Ycoor2, FC, SURF_TYP, SEG_LENGTH, AADT, LNS, BEG_STA, END_STA, SURF_WTH, MED_WTH, StreetLightAll, StreetLightOne, TreeDeb, TreeTrim, Potholes, SEG_ID, CrashCnt, FatalCnt, InjCnt, CrashHour, DayOfWeek, Risk in reader:
            Xcoor1, Ycoor1, Xcoor2, Ycoor2, BEG_STA, END_STA, SEG_LENGTH, AADT, Risk = map(float, (Xcoor1, Ycoor1, Xcoor2, Ycoor2, BEG_STA, END_STA, SEG_LENGTH, AADT, Risk))
            LNS, SURF_WTH, MED_WTH, CrashCnt, FatalCnt, InjCnt, CrashHour, StreetLightAll, StreetLightOne, TreeDeb, TreeTrim, Potholes = map(int, (LNS, SURF_WTH, MED_WTH, CrashCnt, FatalCnt, InjCnt, CrashHour, StreetLightAll, StreetLightOne, TreeDeb, TreeTrim, Potholes))
            FC, SEG_ID, SURF_TYP, DayOfWeek = map(str, (FC, SEG_ID, SURF_TYP, DayOfWeek))
            features.append(Feature(geometry=LineString([(float(Xcoor1), float(Ycoor1)),(float(Xcoor2), float(Ycoor2))]),
                            properties = {
                                    'Function Class': FC,
                                    'Crash Count': CrashCnt,
                                    'Fatals Count': FatalCnt,
                                    'Injury Count': InjCnt,
                                    'Crash Hour': CrashHour,
                                    'Day of Week': DayOfWeek,
                                    'SURF_TYP': SURF_TYP,
                                    'BEG_STA': BEG_STA,
                                    'END_STA': END_STA,
                                    'SEG_LENGTH': SEG_LENGTH,
                                    'SEG_ID': SEG_ID,
                                    'AADT': AADT,
                                    'LNS': LNS,
                                    'SURF_WTH': SURF_WTH,
                                    'MED_WTH': MED_WTH,
                                    'StreetLigthAll': StreetLightAll,
                                    'StreetLightOne': StreetLightOne,
                                    'TreeDeb': TreeDeb,
                                    'TreeTrim': TreeTrim,
                                    'Potholes': Potholes,
                                    'Risk': Risk}))

    collection = FeatureCollection(features)
#     with open('D:/DataScienceProjects/TDI_CAP/OutputData/risk2015.json','w') as f:
#         f.write('%s' %collection)
    with open(json_name,'w') as f:
        f.write('%s' %collection)
    print "Graph saved to geojson!"

#save the graph data to a csv file	
def save_to_csv(G, X_col, cat_col, flo_col, int_col, csv_name):
    X_col = X_col + ['SEG_ID', 'CrashCnt', 'FatalCnt', 'InjCnt', 'CrashHour','DayOfWeek','Risk']
    exclude = ['hr_sin','hr_cos','day_sin','day_cos']
    output_col = [e for e in X_col if e not in exclude]
    int_col = [e for e in int_col if e not in exclude] + ['CrashCnt', 'FatalCnt', 'InjCnt', 'CrashHour']
    cat_col = [e for e in cat_col if e not in exclude] + ['DayOfWeek','SEG_ID']
    flo_col = [e for e in flo_col if e not in exclude] + ['Risk']
    print "Done re-creating variable list."
    
    samples = []
    for edge in G.edges_iter(data=True):
        sample = []
        sample.append(round(edge[0][0],8))
        sample.append(round(edge[0][1],8))
        sample.append(round(edge[1][0],8))
        sample.append(round(edge[1][1],8))

        for var in output_col:
            try:
                sample.append(edge[2][var])
            except:
                continue
        samples.append(sample)
    print "Done creating sample matrix."

    G_df = pd.DataFrame(samples)
    latlon=['Xcoor1','Ycoor1', 'Xcoor2', 'Ycoor2']
    col=latlon+output_col
    G_df.columns=col
	
    #G_df[cat_col] = G_df[cat_col].astype(str)
    #G_df[flo_col] = G_df[flo_col].astype('float64')
    #G_df[latlon] = np.asarray(G_df[latlon])
    #G_df[int_col] = G_df[int_col].astype('int64')
    G_df.to_csv(csv_name, index=False)
    print 'Graph saved to csv file!'