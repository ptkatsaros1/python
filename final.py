import pandas as pd
import re
import numpy as np
import math
from scipy import stats
from statsmodels.formula.api import logit
from statsmodels.formula.api import mnlogit
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from statsmodels.formula.api import ols
from sklearn.cluster import KMeans
import folium
from geopy.geocoders import Nominatim

def main():

    df = pd.read_csv("/Users/tomkatsaros/Documents/testpython/Final Proj 5 Year.csv")
    dfw = pd.read_csv("/Users/tomkatsaros/Documents/testpython/ward_demographics.csv")

    df_ward_violent = count_violent_crimes_ward(df, dfw)

    multi_regression(df_ward_violent)
    line_graph(df)
    pie_chart(df)
    scatter(df, df_ward_violent)
    cluster_analysis_df(df)


    ##merges the original two datasets into the current df dataset
    # merge_df()

def merge_df():
    
    #read in both data sets
    df_crime = pd.read_csv("/Users/tomkatsaros/Documents/testpython/Crimes_-_2001_to_Present.csv")
    df_ward = pd.read_csv("/Users/tomkatsaros/Documents/testpython/ward_demographics.csv")
    # inner join to match demographic data to each crime, based on ward
    df_combo = pd.merge(df_crime, df_ward, on="Ward", how="inner")
    # only read in 5 years of data for performance, look at only pre-covid data
    df_combo = df_combo[df_combo['Year'].between(2014, 2018, inclusive=True)]
    df_combo.to_csv("/Users/tomkatsaros/Desktop/Final Proj 5 Year.csv")

def multi_regression(df_ward_violent):
    print('\n\n')
    print("----------------------------------------------------------------------------------------------------------------------------------------------------")
    print('\n\n')

    x = df_ward_violent[['Income_50000_99999_pct', 'Race_Black_pct']]
    y = df_ward_violent['Count']

    x = sm.add_constant(x)
    model = sm.OLS(y,x).fit()
    predictions = model.predict(x)

    print_model = model.summary()
    print(print_model)

def count_violent_crimes_ward(df, dfw):
    #this df gives the total number of violent crimes for each ward and demographics
    print('\n\n')
    print("----------------------------------------------------------------------------------------------------------------------------------------------------")
    print('\n\n')
    print("VIOLENT CRIME")
    print('\n\n')

    dfv = df[df['Primary Type'].str.contains("HOMICIDE|BATTERY|WEAPONS VIOLATION|ASSAULT|SEX|ROBBERY|HOMICIDE|ARSON|CRIMINAL SEXUAL ASSAULT|HUMAN TRAFFICKING")==True]
    df_ward = dfv.groupby('Ward', as_index=False)['Year'].value_counts()
    df_ward_violent = df_ward.groupby('Ward', as_index=False)['count'].mean()
    df_ward_violent = df_ward_violent.rename({'count':'Count'}, axis=1)
    df_ward_violent = pd.merge(df_ward_violent, dfw, on="Ward", how="inner")
    df_ward_violent = df_ward_violent.drop(['Unnamed: 0'], axis=1)

    return df_ward_violent

def line_graph(df):
    linedf = df
    linedf = linedf.groupby('Primary Type', as_index=False)['Year'].value_counts()

    #line graph violent crime
    df_violent = linedf[linedf['Primary Type'].str.contains("HOMICIDE|BATTERY|WEAPONS VIOLATION|ASSAULT|SEX|ROBBERY|HOMICIDE|ARSON|CRIMINAL SEXUAL ASSAULT|HUMAN TRAFFICKING")==True]
    total_violent = df_violent.groupby('Year', as_index=False)['count'].sum()
    total_violent = total_violent.sort_values(by=['Year'])

    # violent crime line
    plt.figure()
    plt.plot(total_violent['Year'], total_violent['count'], color = 'black', marker = 'o')
    plt.title('Violent Crime Line Graph')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.xticks(np.arange(2014,2019,1))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def pie_chart(df):
    df_bool=df
    violent_crimes="HOMICIDE|BATTERY|WEAPONS VIOLATION|ASSAULT|SEX|ROBBERY|HOMICIDE|ARSON|CRIMINAL SEXUAL ASSAULT|HUMAN TRAFFICKING"
    df_bool['violent_non'] = 'non-violent'
    df_bool.loc[df_bool['Primary Type'].str.contains(violent_crimes), 'violent_non'] = 'violent'
    labels1 = 'non-violent crimes', 'violent crimes'
    pie_data = (df['violent_non'].value_counts(normalize=True) * 100)

    explode = (0.1, 0)  
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_data, explode=explode, labels=labels1, autopct='%1.1f%%',
            shadow=True, startangle=90, labeldistance=1.2)
    ax1.axis('equal')
    plt.show()

def scatter(df, df_ward_violent):

    # violent crime vs low income % scatter
    plt.figure()
    plt.scatter(df_ward_violent['Income_24999_minus_pct'], df_ward_violent['Count'], color = 'black', marker = 'o')
    plt.title('Violent Crime vs % of Low-Income  Residents Scatter Plot')
    plt.xlabel('Low-Income Residents %')
    plt.ylabel('Count')
    x = df_ward_violent['Income_24999_minus_pct']
    a, b = np.polyfit(df_ward_violent['Income_24999_minus_pct'], df_ward_violent['Count'], 1)
    plt.plot(x, a*x+b, color='red', linestyle='dashdot', linewidth=2)
    correlation2, p_value2 = stats.pearsonr(df_ward_violent['Income_24999_minus_pct'], df_ward_violent['Count']) 
    plt.text(10, 4400, f'Correlation: {correlation2.round(4)}', fontsize = 10)
    plt.text(12, 4200, f'P-Value: {p_value2.round(4)}', fontsize = 10)
    plt.tight_layout()
    plt.show()

    # violent crime vs medium income % scatter
    plt.figure()
    plt.scatter(df_ward_violent['Income_50000_99999_pct'], df_ward_violent['Count'], color = 'black', marker = 'o')
    plt.title('Violent Crime vs % of Middle Class Residents Plot')
    plt.xlabel('% of Middle Class Residents')
    plt.ylabel('Count')
    x = df_ward_violent['Income_50000_99999_pct']
    a, b = np.polyfit(df_ward_violent['Income_50000_99999_pct'], df_ward_violent['Count'], 1)
    plt.plot(x, a*x+b, color='red', linestyle='dashdot', linewidth=2)
    correlation1, p_value1 = stats.pearsonr(df_ward_violent['Income_50000_99999_pct'], df_ward_violent['Count']) 
    plt.text(29, 4400, f'Correlation: {correlation1.round(4)}', fontsize = 10)
    plt.text(31, 4200, f'P-Value: {p_value1.round(4)}', fontsize = 10)
    plt.tight_layout()
    plt.show()

    # violent crime vs white scatter
    plt.figure()
    plt.scatter(df_ward_violent['Race_White_pct'], df_ward_violent['Count'], color = 'black', marker = 'o')
    plt.title('Violent Crime vs White % Scatter Plot')
    plt.xlabel('White %')
    plt.ylabel('Count')
    x = df_ward_violent['Race_White_pct']
    a, b = np.polyfit(df_ward_violent['Race_White_pct'], df_ward_violent['Count'], 1)
    plt.plot(x, a*x+b, color='red', linestyle='dashdot', linewidth=2)
    correlationw, p_valuew = stats.pearsonr(df_ward_violent['Race_White_pct'], df_ward_violent['Count']) 
    plt.text(63, 4400, f'Correlation: {correlationw.round(4)}', fontsize = 10)
    plt.text(65, 4200, f'P-Value: {p_valuew.round(4)}', fontsize = 10)
    plt.tight_layout()
    plt.show()

    # violent crime vs black scatter
    plt.figure()
    plt.scatter(df_ward_violent['Race_Black_pct'], df_ward_violent['Count'], color = 'black', marker = 'o')
    plt.title('Violent Crime vs Black % Scatter Plot')
    plt.xlabel('Black %')
    plt.ylabel('Count')
    x = df_ward_violent['Race_Black_pct']
    a, b = np.polyfit(df_ward_violent['Race_Black_pct'], df_ward_violent['Count'], 1)
    plt.plot(x, a*x+b, color='red', linestyle='dashdot', linewidth=2)
    correlationb, p_valueb = stats.pearsonr(df_ward_violent['Race_Black_pct'], df_ward_violent['Count']) 
    plt.text(0, 4400, f'Correlation: {correlationb.round(4)}', fontsize = 10)
    plt.text(3, 4200, f'P-Value: {p_valueb.round(4)}', fontsize = 10)
    plt.tight_layout()
    plt.show()

def create_cluster_map(df, centroid_lat_long_list, weight, color_list):
    centroid_actual_lat_long_list = []

    for lat_long in centroid_lat_long_list:
        lat_actual = lat_long[0] * df['Latitude'].std() / weight + df['Latitude'].mean()
        long_actual = lat_long[1] * df['Longitude'].std() / weight + df['Longitude'].mean()
        centroid_actual_lat_long_list.append([lat_actual, long_actual])

    # tiles options: Stamen Toner, Stamen Terrain, Stamen Water Color, cartodbpositron, cartodbdark_matter
    cluster_map = folium.Map(location = [df.Latitude.mean(), df.Longitude.mean()], tiles='Stamen Terrain')

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row.Latitude,row.Longitude],
            radius=0.1,
            color=row.cluster_color,
            fill=False,
            fill_color=row.cluster_color,
            popup=folium.Popup(str(row.Latitude) + ', ' + str(row.Longitude))).add_to(cluster_map)

    geolocator = Nominatim(user_agent="geoapiExercises")

    for i in range(len(centroid_actual_lat_long_list)):
        lat_long = centroid_actual_lat_long_list[i]
        cluster_color = color_list[i]
        cluster_stats = dict(df[df['cluster'] == i][['Race-Black_pct', 'Race-White_pct', 'Income-0-49999_pct', 'category']].mean().round(2))
        location = geolocator.reverse(str(lat_long[0])+","+str(lat_long[1]))
        address_all = location.raw['address']
        address = ''
        if address_all.get('road') is not None:
            address += address_all.get('road') + ', '
        if address_all.get('city') is not None:
            address += address_all.get('city') + ', '
        if address_all.get('town') is not None:
            address += address_all.get('town') + ', '
        if address_all.get('postcode') is not None:
            address += str(address_all.get('postcode')) + ', '
        if address_all.get('county') is not None:
            address += address_all.get('county')
        
        folium.Marker(
            location=[lat_long[0],
            lat_long[1]],
            radius = 5,
            fill_opacity=0.5, 
            popup=folium.Popup(f'###Cluster color: {cluster_color}### ###Avg Stats:{cluster_stats}### ###Geographical Centroid:{address}###'),
            icon=folium.Icon(color=color_list[i], icon='building',prefix='fa')).add_to(cluster_map)

    cluster_map.save('cluster_map.html')
    
def get_centroid_lat_long_list(centroid_list):
    centroid_lat_long_list = []
    
    for center in centroid_list:
        centroid_lat_long_list.append([center[len(centroid_list)-2], center[len(centroid_list)-1]])
    
    return centroid_lat_long_list
    
def find_cluster_count(df, variable_list):
    wcss = []
    
    for i in range(1, 16):
        k_means_model = KMeans(n_clusters=i)
        k_means_model.fit(df[variable_list])
        wcss.append(k_means_model.inertia_)
        
    plt.figure()
    plt.plot(range(1,16), wcss, marker='*')
    plt.title('Elbow Curve')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    
def plot_clusters(df, cluster_count, centroid_list, color_list):
   
    
    fig = plt.figure()
    fig_3d = fig.add_subplot(projection='3d')
    
    for i in range(cluster_count):
        df_cluster = df[df['cluster'] == i]
        fig_3d.scatter(df_cluster['Race-Black_pct_norm'], 
                       df_cluster['Race-White_pct_norm'], 
                       df_cluster['Income-0-49999_pct_norm'], 
                       s=1, color=color_list[i])
        # Add the ceontroids of the clusters here
        fig_3d.scatter(centroid_list[i][0], centroid_list[i][1], centroid_list[i][2], color='black', s=30)

    fig_3d.set_title('Clusters')
    fig_3d.set_xlabel('Black Population')
    fig_3d.set_ylabel('White Population')
    fig_3d.set_zlabel('Income')
    plt.show()
    
    
def get_cluster_color(cluster, color_list):
    return color_list[cluster]
    
    
def create_clusters(df, variable_list, cluster_count, color_list):
    k_means_model = KMeans(n_clusters=cluster_count)
    df['cluster'] = k_means_model.fit_predict(df[variable_list])
    df['cluster_color'] = df.apply(lambda x: get_cluster_color(x.cluster, color_list), axis=1)
    centroid_list = k_means_model.cluster_centers_
    
    df.to_excel('clustered_data.xlsx', index=False)    
    return df, centroid_list

def get_category(primary_type, type_list):
    if primary_type in type_list[0]:
        return 0
    else:
        return 1

def cluster_analysis_df(df):
    cluster_df = df[df['Primary Type'].str.contains("HOMICIDE|BATTERY|WEAPONS VIOLATION|ASSAULT|SEX|ROBBERY|ARSON|CRIMINAL SEXUAL ASSAULT|HUMAN TRAFFICKING")==True]
    cluster_df = cluster_df.drop(['Unnamed: 0.1', 'Case Number', 'Date', 'Block', 'IUCR', 'Description', 'Location Description',
    'Beat', 'District', 'Community Area', 'FBI Code','X Coordinate', 'Y Coordinate', 'Updated On', 'Unnamed: 0'], axis=1)
    
    cluster_df = cluster_df[cluster_df['Year'] == 2018 ]
    
    cluster_df['Longitude'] = pd.to_numeric(cluster_df['Longitude'])
    cluster_df['Latitude'] = pd.to_numeric(cluster_df['Latitude'])
    cluster_df['Income-0-49999_pct'] = cluster_df['Income-24999_minus_pct'] + cluster_df['Income-25000-49999_pct']
    cluster_df = cluster_df.dropna()
    
    type_list = [["ARSON", "ROBBERY", "WEAPONS VIOLATION"], 
                 ["ASSAULT", "BATTERY", "CRIM SEXUAL ASSAULT", "CRIMINAL SEXUAL ASSAULT", "SEX OFFENSE", "HOMICIDE", "HUMAN TRAFFICKING"]]
    
    cluster_df['category'] = df.apply(lambda x: get_category(x['Primary Type'], type_list), axis=1)
    
    #add here to include in cluster
    
    var_weight_dict = {'Race-Black_pct': 1, 'Race-White_pct': 1, 'Income-0-49999_pct': 1, 
                       'category': 1,
                       'Latitude': 1.5, 'Longitude': 1.5}
    var_norm_list = []

    for var, weight in var_weight_dict.items():
        if var != "category":
            cluster_df[f'{var}_norm'] = weight * (cluster_df[var] - cluster_df[var].mean())/cluster_df[var].std()
        else:
            cluster_df[f'{var}_norm'] = cluster_df[var]
        var_norm_list.append(f'{var}_norm')
        
    var_norm_list_use = var_norm_list
    
    find_cluster_count(cluster_df, var_norm_list_use)
    
    cluster_count = 4

    color_list = ['red', 'blue', 'green', 
                  'purple', 'orange', 'pink', 
                  'cyan', 'grey', 'brown']
    
    cluster_df, centroid_list = create_clusters(cluster_df, var_norm_list_use, cluster_count, color_list)
    
    plot_clusters(cluster_df, cluster_count, centroid_list, color_list)
    
    centroid_lat_long_list = get_centroid_lat_long_list(centroid_list)
    
    create_cluster_map(cluster_df, centroid_lat_long_list, var_weight_dict['Latitude'], color_list)
    
    # print(centroid_list)
    
    # print(cluster_df)


main()







