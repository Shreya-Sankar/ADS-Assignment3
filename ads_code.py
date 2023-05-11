# -*- coding: utf-8 -*-
"""
Created on Tue May  9 23:01:36 2023

@author: Shreya Thekkiniyedath Kudalvalli
"""

#Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn import cluster
import errors as err


#Read and clean the datafile
def read_file(fn):
    """
    Reads a CSV file and returns a pandas DataFrame.

    Parameters:
    ------------    
    fn (str): The filename of the CSV file to be read.

    Returns:
    ---------    
    df (pandas.DatFrame): The DataFrame containing the data 
    read from the CSV file.
    """
    address = "C:\\Users\\user\\OneDrive\\ASD1 Assignment3\\" + fn
    df = pd.read_csv(address)
    df = df.drop(df.columns[:2], axis=1)
    df=df.drop(columns=['Country Code'])
    
    # Remove the string of year from column names
    df.columns = df.columns.str.replace(' \[YR\d{4}\]', '', regex=True)
    countries=['Japan','India']
    country_code=['JPN','IND']
    
    #Transpose the dataframe
    df = df[df['Country Name'].isin(countries)].T 
    #Rename columns
    df = df.rename({'Country Name': 'year'})
    df = df.reset_index().rename(columns={'index': 'year'})
    
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.replace('..', np.nan)
    df = df.replace(np.nan,0)
    
    df["year"] = df["year"].astype(int)
    df["India"]=df["India"].astype(float)
    df["Japan"]=df["Japan"].astype(float)
    
    return df



def curve_fun(t, scale, growth):
 """
 
 Parameters
 ----------
 t : TYPE
 List of values
 scale : TYPE
 Scale of curve.
 growth : TYPE
 Growth of the curve.
 Returns
 -------
 c : TYPE
 Result
 """
 c = scale * np.exp(growth * (t-1990))
 return c


def gdp(df_gdp):
  """
    plots the GDP per capita for Japan and India.

    Parameters:
    ------------
    df_gdp(pandas.DataFrame): A DataFrame containing 
    the data of GDP per capita.

    Returns:
    ---------
    None
   """

  plt.plot(df_gdp["year"], df_gdp["India"], c="crimson", label="India")
  plt.plot(df_gdp["year"], df_gdp["Japan"], c="darkviolet", label="Japan")
  plt.xlim(1990,2019)
  plt.xlabel("Year")
  plt.ylabel(" GDP per capita growth (annual %)")
  plt.legend()
  plt.title("GDP per capita for india and japan")
  plt.savefig("GDP.png", dpi = 300, bbox_inches='tight')
  plt.show()
    

#Calling the file read function
df_co2=read_file("Co2_Emissions.csv")
df_gdp=read_file("GDP_per_Capita.csv")
df_renew= read_file("Renewable_Energy.csv")

#Performing curve fit for India

param, cov = opt.curve_fit(
    curve_fun,
    df_co2["year"],
    df_co2["India"],
    p0=[4e8, 0.1]
    )

#  np.sqrt() is used to return the 
# non-negetive square root of an array element-wise.

# np.diag() will extract or construct a diagonal array.
sigma = np.sqrt(np.diag(cov))

#Error
low,up = err.err_ranges(df_co2["year"],curve_fun,param,sigma)
df_co2["fit_value"] = curve_fun(df_co2["year"], * param)

# 1: Plotting the co2 emission values for India
plt.figure()
plt.title("CO2 emissions (metric tons per capita) - India")
plt.plot(df_co2["year"],df_co2["India"], c="crimson", label="data")
plt.plot(df_co2["year"],df_co2["fit_value"],c="darkviolet",label="fit")
plt.fill_between(df_co2["year"],low,up,alpha=0.4)
plt.legend()
plt.xlim(1990,2019)
plt.xlabel("Year")
plt.ylabel("CO2")
plt.savefig("Co2_India.png", dpi = 300, bbox_inches='tight')
plt.show()



# 2: Plotting the predicted values for India co2 emission
plt.figure()
plt.title("CO2 emission prediction of India")
pred_year = np.arange(1990,2030)
pred_ind = curve_fun(pred_year,*param)
plt.plot(df_co2["year"],df_co2["India"],c="crimson", label="data")
plt.plot(
    pred_year,
    pred_ind,
    c="darkviolet", 
    label="predicted values"
    )
plt.legend()
plt.xlabel("Year")
plt.ylabel("CO2")
plt.savefig("Co2_India_Predicted.png", dpi = 300, bbox_inches='tight')
plt.show()



#Curve fit for Japan
param, cov = opt.curve_fit(
    curve_fun,
    df_co2["year"],
    df_co2["Japan"],
    p0=[4e8, 0.1]
    )
sigma = np.sqrt(np.diag(cov))
low,up = err.err_ranges(df_co2["year"],curve_fun,param,sigma)
df_co2["fit_value"] = curve_fun(df_co2["year"], * param)


# 3: Plotting co2 emission prediction for Japan
plt.figure()
plt.title("Japan CO2 emission prediction For 2030")
pred_year = np.arange(1990,2030)
pred_ind = curve_fun(pred_year,*param)
plt.plot(df_co2["year"],df_co2["Japan"],c="crimson", label="data")
plt.plot(pred_year,pred_ind, c="darkviolet", label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("CO2")
plt.savefig("Co2_Japan_Predicted.png", dpi = 300, bbox_inches='tight')
plt.show()



# 4: Renewable energy use as a percentage of total energy - India
#Curve fit for India
param, cov = opt.curve_fit(
    curve_fun,
    df_renew["year"],
    df_renew["India"],
    p0=[4e8, 0.1]
    )
sigma = np.sqrt(np.diag(cov))
low,up = err.err_ranges(df_renew["year"],curve_fun,param,sigma)
df_renew["fit_value"] = curve_fun(df_renew["year"], * param)

plt.figure()
plt.title("Renewable energy use as a percentage of total energy - India")
plt.plot(
    df_renew["year"],
    df_renew["India"], 
    c="crimson", 
    label="data"
    )
plt.plot(
    df_renew["year"],
    df_renew["fit_value"], 
    c="darkviolet", 
    label="fit"
    )
plt.fill_between(df_renew["year"],low,up,alpha=0.3)
plt.legend()
plt.xlim(1990,2019)
plt.xlabel("Year")
plt.ylabel("Renewable energy(% of total energy use)")
plt.savefig("Renewable_India.png", dpi = 300, bbox_inches='tight')
plt.show()


#5) Renewable energy prediction - India

plt.figure()
plt.title("Renewable energy prediction - India")
pred_year = np.arange(1990,2030)
pred_ind = curve_fun(pred_year,*param)
plt.plot(df_renew["year"],df_renew["India"],c="crimson", label="data")
plt.plot(pred_year,pred_ind, c="darkviolet", label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Renewable energy(% of total energy use)")
plt.savefig(
    "Renewable_energy_prediction_India.png", 
    dpi = 300, 
    bbox_inches='tight'
    )
plt.show()

#6) Renewable energy prediction - Japan

param, cov = opt.curve_fit(
    curve_fun,
    df_renew["year"],
    df_renew["Japan"],
    p0=[4e8, 0.1]
    )
sigma = np.sqrt(np.diag(cov))
low,up = err.err_ranges(df_renew["year"],curve_fun,param,sigma)
df_renew["fit_value"] = curve_fun(df_renew["year"], * param)


plt.figure()
plt.title("Renewable energy prediction - Japan")
pred_year = np.arange(1990,2030)
pred_ind = curve_fun(pred_year,*param)
plt.plot(df_renew["year"],df_renew["Japan"], c="crimson", label="data")
plt.plot(pred_year,pred_ind, c="darkviolet", label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Renewable energy(% of total energy use)")
plt.savefig(
    "Renewable_Prediction_Japan.png",
    dpi = 300, 
    bbox_inches='tight'
    )
plt.show()

# 7) Gdp per capita

#Calling the gdp() for visualizing gdp per capita of India and Japan
gdp(df_gdp)


# 8) Japan and India - CO2 Emission

df_co2= df_co2.iloc[:,1:3]
#Normalize data
df_co2_norm=(df_co2 - df_co2.mean()) / df_co2.std()
df_renew_norm=(df_renew - df_renew.mean()) / df_renew.std()

#Create cluster and visualize co2 emissions of given countries
# Kmeans Clustering

kmean = cluster.KMeans(n_clusters=4).fit(df_co2_norm)
label = kmean.labels_
plt.scatter(df_co2_norm["Japan"],df_co2_norm["India"],c=label,cmap="coolwarm")
plt.title("Japan and India - CO2 Emission")
plt.xlabel("Co2 emission of Japan")
plt.ylabel("co2 emission of India")
c = kmean.cluster_centers_
plt.savefig("Scatter_Japan_India_CO2.png", 
            dpi = 300, 
            bbox_inches='tight'
            )
plt.show()

#9) co2 emission vs renewable enery usage - India

india = pd.DataFrame()
india["co2_emission"] = df_co2_norm["India"]
india["renewable_energy"] = df_renew_norm["India"]

kmean = cluster.KMeans(n_clusters=4).fit(india)
label = kmean.labels_
plt.scatter(india["co2_emission"], india["renewable_energy"],
            c=label,cmap="coolwarm")
plt.title("co2 emission vs renewable enery usage - India")
plt.xlabel("co2 emission")
plt.ylabel("Renewable energy")

plt.savefig(
    "Scatter_CO2_vs_Renewable_India.png", 
    dpi = 300,
    bbox_inches='tight'
    )
c = kmean.cluster_centers_

for t in range(2):
 xc,yc = c[t,:]
 plt.plot(xc,yc,"ok",markersize=8)
plt.figure()

plt.show()