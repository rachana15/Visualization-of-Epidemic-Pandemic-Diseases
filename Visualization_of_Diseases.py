#!/usr/bin/env python
# coding: utf-8

# In[30]:


get_ipython().system('pip install pycountry_convert ')
get_ipython().system('pip install folium')
get_ipython().system('pip install calmap')
get_ipython().system('wget https://raw.githubusercontent.com/tarunk04/COVID-19-CaseStudy-and-Predictions/master/models/model_deaths_v2.h5')
get_ipython().system('wget https://raw.githubusercontent.com/tarunk04/COVID-19-CaseStudy-and-Predictions/master/models/model_confirmed_v2.h5')
get_ipython().system('wget https://raw.githubusercontent.com/tarunk04/COVID-19-CaseStudy-and-Predictions/master/models/model_usa_c.h5')



# In[31]:


pip install plotly


# In[32]:


pip install keras


# In[33]:


pip install tensorflow


# In[353]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker 
import pycountry_convert as pc
import folium
import branca
from datetime import datetime, timedelta,date
from scipy.interpolate import make_interp_spline, BSpline
import plotly.express as px
import json, requests
import calmap

from keras.layers import Input, Dense, Activation, LeakyReLU
from keras import models
from keras.optimizers import RMSprop, Adam

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[354]:


df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')


# In[355]:


#COVID Data
# df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
df_table = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv",parse_dates=['Last_Update'])


# In[356]:


df_covid19 = df_covid19.drop(["People_Tested","People_Hospitalized","UID","ISO3","Mortality_Rate"],axis =1)
# df_covid19.head(2)


# In[357]:


# df_confirmed.head(2)


# In[358]:


df_confirmed = df_confirmed.rename(columns={"Province/State":"state","Country/Region": "country"})
df_deaths = df_deaths.rename(columns={"Province/State":"state","Country/Region": "country"})
df_covid19 = df_covid19.rename(columns={"Country_Region": "country"})
df_covid19["Active"] = df_covid19["Confirmed"]-df_covid19["Recovered"]-df_covid19["Deaths"]
# df_recovered = df_recovered.rename(columns={"Province/State":"state","Country/Region": "country"})


# In[359]:


# Changing the conuntry names as required by pycountry_convert Lib
df_confirmed.loc[df_confirmed['country'] == "US", "country"] = "USA"
df_deaths.loc[df_deaths['country'] == "US", "country"] = "USA"
df_covid19.loc[df_covid19['country'] == "US", "country"] = "USA"
df_table.loc[df_table['Country_Region'] == "US", "Country_Region"] = "USA"
# df_recovered.loc[df_recovered['country'] == "US", "country"] = "USA"


df_confirmed.loc[df_confirmed['country'] == 'Korea, South', "country"] = 'South Korea'
df_deaths.loc[df_deaths['country'] == 'Korea, South', "country"] = 'South Korea'
df_covid19.loc[df_covid19['country'] == "Korea, South", "country"] = "South Korea"
df_table.loc[df_table['Country_Region'] == "Korea, South", "Country_Region"] = "South Korea"
# df_recovered.loc[df_recovered['country'] == 'Korea, South', "country"] = 'South Korea'

df_confirmed.loc[df_confirmed['country'] == 'Taiwan*', "country"] = 'Taiwan'
df_deaths.loc[df_deaths['country'] == 'Taiwan*', "country"] = 'Taiwan'
df_covid19.loc[df_covid19['country'] == "Taiwan*", "country"] = "Taiwan"
df_table.loc[df_table['Country_Region'] == "Taiwan*", "Country_Region"] = "Taiwan"
# df_recovered.loc[df_recovered['country'] == 'Taiwan*', "country"] = 'Taiwan'

df_confirmed.loc[df_confirmed['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'
df_deaths.loc[df_deaths['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'
df_covid19.loc[df_covid19['country'] == "Congo (Kinshasa)", "country"] = "Democratic Republic of the Congo"
df_table.loc[df_table['Country_Region'] == "Congo (Kinshasa)", "Country_Region"] = "Democratic Republic of the Congo"
# df_recovered.loc[df_recovered['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'

df_confirmed.loc[df_confirmed['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"
df_deaths.loc[df_deaths['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"
df_covid19.loc[df_covid19['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"
df_table.loc[df_table['Country_Region'] == "Cote d'Ivoire", "Country_Region"] = "Côte d'Ivoire"
# df_recovered.loc[df_recovered['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"

df_confirmed.loc[df_confirmed['country'] == "Reunion", "country"] = "Réunion"
df_deaths.loc[df_deaths['country'] == "Reunion", "country"] = "Réunion"
df_covid19.loc[df_covid19['country'] == "Reunion", "country"] = "Réunion"
df_table.loc[df_table['Country_Region'] == "Reunion", "Country_Region"] = "Réunion"
# df_recovered.loc[df_recovered['country'] == "Reunion", "country"] = "Réunion"

df_confirmed.loc[df_confirmed['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'
df_deaths.loc[df_deaths['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'
df_covid19.loc[df_covid19['country'] == "Congo (Brazzaville)", "country"] = "Republic of the Congo"
df_table.loc[df_table['Country_Region'] == "Congo (Brazzaville)", "Country_Region"] = "Republic of the Congo"
# df_recovered.loc[df_recovered['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'

df_confirmed.loc[df_confirmed['country'] == 'Bahamas, The', "country"] = 'Bahamas'
df_deaths.loc[df_deaths['country'] == 'Bahamas, The', "country"] = 'Bahamas'
df_covid19.loc[df_covid19['country'] == "Bahamas, The", "country"] = "Bahamas"
df_table.loc[df_table['Country_Region'] == "Bahamas, The", "Country_Region"] = "Bahamas"
# df_recovered.loc[df_recovered['country'] == 'Bahamas, The', "country"] = 'Bahamas'

df_confirmed.loc[df_confirmed['country'] == 'Gambia, The', "country"] = 'Gambia'
df_deaths.loc[df_deaths['country'] == 'Gambia, The', "country"] = 'Gambia'
df_covid19.loc[df_covid19['country'] == "Gambia, The", "country"] = "Gambia"
df_table.loc[df_table['Country_Region'] == "Gambia", "Country_Region"] = "Gambia"
# df_recovered.loc[df_recovered['country'] == 'Gambia, The', "country"] = 'Gambia'

# getting all countries
countries = np.asarray(df_confirmed["country"])
countries1 = np.asarray(df_covid19["country"])
# Continent_code to Continent_names
continents = {
    'NA': 'North America',
    'SA': 'South America', 
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
    'EU' : 'Europe',
    'na' : 'Others'
}

# Defininng Function for getting continent code for country.
def country_to_continent_code(country):
    try:
        return pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))
    except :
        return 'na'

#Collecting Continent Information
df_confirmed.insert(2,"continent", [continents[country_to_continent_code(country)] for country in countries[:]])
df_deaths.insert(2,"continent",  [continents[country_to_continent_code(country)] for country in countries[:]])
df_covid19.insert(1,"continent",  [continents[country_to_continent_code(country)] for country in countries1[:]])
df_table.insert(1,"continent",  [continents[country_to_continent_code(country)] for country in df_table["Country_Region"].values])
# df_recovered.insert(2,"continent",  [continents[country_to_continent_code(country)] for country in countries[:]] )   


# In[360]:


df_confirmed = df_confirmed.replace(np.nan, '', regex=True)
df_deaths = df_deaths.replace(np.nan, '', regex=True)


# In[361]:


df_countries_cases = df_covid19.copy().drop(['Lat','Long_','continent','Last_Update'],axis =1)
df_countries_cases.index = df_countries_cases["country"]
df_countries_cases = df_countries_cases.drop(['country'],axis=1)

df_continents_cases = df_covid19.copy().drop(['Lat','Long_','country','Last_Update'],axis =1)
df_continents_cases = df_continents_cases.groupby(["continent"]).sum()

df_countries_cases.fillna(0,inplace=True)
df_continents_cases.fillna(0,inplace=True)


# In[362]:


df_t = pd.DataFrame(pd.to_numeric(df_countries_cases.sum()),dtype=np.float64).transpose()
print(df_t)
df_t["Mortality Rate (per 100)"] = np.round(100*df_t["Deaths"]/df_t["Confirmed"],2)
df_t.style.background_gradient(cmap='Wistia',axis=1).format("{:.0f}",subset=["Confirmed"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[363]:


df_countries_cases = df_covid19.copy().drop(['Lat','Long_','continent','Last_Update'],axis =1)
df_countries_cases.index = df_countries_cases["country"]
df_countries_cases = df_countries_cases.drop(['country'],axis=1)

df_continents_cases = df_covid19.copy().drop(['Lat','Long_','country','Last_Update'],axis =1)
df_continents_cases = df_continents_cases.groupby(["continent"]).sum()

df_countries_cases.fillna(0,inplace=True)
df_continents_cases.fillna(0,inplace=True)


# In[364]:


temp_df = pd.DataFrame(df_countries_cases['Confirmed'])
temp_df = temp_df.reset_index()
fig = px.choropleth(temp_df, locations="country",
                    color=np.log10(temp_df["Confirmed"]), # lifeExp is a column of gapminder
                    hover_name="country", # column to add to hover information
                    hover_data=["Confirmed"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Confirmed Cases for COVID-19 Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="Confirmed Cases(Log Scale)",colorscale="Reds")
# fig.to_image("Global Heat Map confirmed.png")
fig.show()


# In[365]:


temp_df = pd.DataFrame(df_countries_cases['Deaths'])
temp_df = temp_df.reset_index()
fig = px.choropleth(temp_df, locations="country",
                    color=np.log10(temp_df["Deaths"]), # lifeExp is a column of gapminder
                    hover_name="country", # column to add to hover information
                    hover_data=["Deaths"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Deaths Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="Deaths(Log Scale)",colorscale="Reds")
# fig.to_image("Global Heat Map confirmed.png")
fig.show()


# In[366]:


f = plt.figure(figsize=(10,5))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Confirmed')["Confirmed"].index[-10:],df_countries_cases.sort_values('Confirmed')["Confirmed"].values[-10:],color="darkcyan")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Confirmed Cases",fontsize=18)
plt.title("Top 10 Countries (Confirmed Cases)",fontsize=20)
plt.grid(alpha=0.3)
# plt.savefig(out+'Top 10 Countries (Confirmed Cases).png')


# In[373]:


c_lat = df_covid19.copy().drop(['Lat','Long_','continent','Last_Update'],axis =1)
c_lat = c_lat.groupby('country')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

fig = px.treemap(c_lat.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 
                 path=["country"], values="Confirmed", title='COVID-19',
                 color_discrete_sequence = px.colors.qualitative.Dark2)
fig.data[0].textinfo = 'label+text+value'
fig.show()


# In[ ]:





# In[ ]:





# In[248]:


# EBOLA
# ------

# ebola dataset
ebola_14 = pd.read_csv("/Users/rachanabhaskar/Desktop/paper/Data/ebola.csv", 
                       parse_dates=['Date'])

# ebola_14 = ebola_14[ebola_14['Date']!=max(ebola_14['Date'])]

# selecting important columns only
ebola_14 = ebola_14[['Date', 'Country', 'No. of confirmed, probable and suspected cases',
                     'No. of confirmed, probable and suspected deaths']]

# renaming columns
ebola_14.columns = ['Date', 'Country', 'Cases', 'Deaths']
ebola_14.head()

# group by date and country
ebola_14 = ebola_14.groupby(['Date', 'Country'])['Cases', 'Deaths']
ebola_14 = ebola_14.sum().reset_index()

# filling missing values
ebola_14['Cases'] = ebola_14['Cases'].fillna(0)
ebola_14['Deaths'] = ebola_14['Deaths'].fillna(0)

# converting datatypes
ebola_14['Cases'] = ebola_14['Cases'].astype('int')
ebola_14['Deaths'] = ebola_14['Deaths'].astype('int')

# latest
e_lat = ebola_14[ebola_14['Date'] == max(ebola_14['Date'])].reset_index()

# latest grouped by country
e_lat_grp = e_lat.groupby('Country')['Cases', 'Deaths'].sum().reset_index()

# nth day
ebola_14['nth_day'] = (ebola_14['Date'] - min(ebola_14['Date'])).dt.days

# day by day
e_dbd = ebola_14.groupby('Date')['Cases', 'Deaths'].sum().reset_index()

# nth day
e_dbd['nth_day'] = ebola_14.groupby('Date')['nth_day'].max().values

# no. of countries
temp = ebola_14[ebola_14['Cases']>0]
e_dbd['n_countries'] = temp.groupby('Date')['Country'].apply(len).values

e_dbd['new_cases'] = e_dbd['Cases'].diff()
e_dbd['new_deaths'] = e_dbd['Deaths'].diff()
e_dbd['epidemic'] = 'EBOLA'

ebola_14.head()


# In[249]:


# df_countries_cases_ebola = ebola_14.copy().drop(['Date'],axis =1)
# df_countries_cases_ebola.index = df_countries_cases_ebola["Country"]
# df_countries_cases_ebola = df_countries_cases_ebola.drop(['Country'],axis=1)
# df_countries_cases_ebola.fillna(0,inplace=True)


# In[250]:


# print(e_dbd)


# In[251]:


e_dbd.sum()
mortality_rate = e_dbd['new_cases'].sum() /e_dbd['new_deaths'].sum()
print(mortality_rate)


# In[252]:


fig = px.choropleth(e_lat_grp, locations="Country", locationmode='country names',
                    color=np.log10(e_lat_grp["Cases"]), hover_name="Country", hover_data = ["Cases"],
                    color_continuous_scale=px.colors.sequential.Plasma, title='EBOLA 2014')
# fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Confirmed Cases for EBOLA Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="Confirmed Cases(Log Scale)",colorscale="Reds")
fig.update(layout_coloraxis_showscale=True)
fig.show()


# In[253]:


fig = px.choropleth(e_lat_grp, locations="Country", locationmode='country names',
                    color=np.log10(e_lat_grp["Deaths"]), hover_name="Country", hover_data = ["Deaths"],
                    color_continuous_scale=px.colors.sequential.Plasma, title='EBOLA 2014')
# fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Death Cases Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="Death Cases(Log Scale)",colorscale="Reds")
fig.update(layout_coloraxis_showscale=True)
fig.show()


# In[254]:


f = plt.figure(figsize=(10,5))
f.add_subplot(111)
plt.axes(axisbelow=True)
plt.barh(e_lat_grp.sort_values('Cases')["Country"].values[-10:],e_lat_grp.sort_values('Cases')["Cases"].index[-10:],color="darkcyan")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Confirmed Cases",fontsize=18)
plt.title("Top 10 Countries (Confirmed Cases)",fontsize=20)
plt.grid(alpha=0.3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[255]:


# sars dataset
sars_03 = pd.read_csv("/Users/rachanabhaskar/Desktop/paper/Data/sars_2003_complete_dataset_clean.csv", 
                       parse_dates=['Date'])

# selecting important columns only
sars_03 = sars_03[['Date', 'Country', 'Cumulative number of case(s)', 
                   'Number of deaths', 'Number recovered']]

# renaming columns
sars_03.columns = ['Date', 'Country', 'Cases', 'Deaths', 'Recovered']

# group by date and country
sars_03 = sars_03.groupby(['Date', 'Country'])['Cases', 'Deaths', 'Recovered']
sars_03 = sars_03.sum().reset_index()

# latest
s_lat = sars_03[sars_03['Date'] == max(sars_03['Date'])].reset_index()

# latest grouped by country
s_lat_grp = s_lat.groupby('Country')['Cases', 'Deaths', 'Recovered'].sum().reset_index()

# nth day
sars_03['nth_day'] = (sars_03['Date'] - min(sars_03['Date'])).dt.days

# day by day
s_dbd = sars_03.groupby('Date')['Cases', 'Deaths', 'Recovered'].sum().reset_index()

# nth day
s_dbd['nth_day'] = sars_03.groupby('Date')['nth_day'].max().values

# no. of countries
temp = sars_03[sars_03['Cases']>0]
s_dbd['n_countries'] = temp.groupby('Date')['Country'].apply(len).values


s_dbd['new_cases'] = s_dbd['Cases'].diff()
s_dbd['new_deaths'] = s_dbd['Deaths'].diff()
s_dbd['epidemic'] = 'SARS'

# s_dbd.head()


# In[256]:


fig = px.choropleth(s_lat_grp, locations="Country", locationmode='country names',
                    color=np.log10(s_lat_grp["Cases"]), hover_name="Country", hover_data = ["Cases"], 
                    color_continuous_scale="Sunsetdark", title='SARS 2003')
fig.update(layout_coloraxis_showscale=True)
fig.update_layout(title_text="Confirmed Cases for SARS Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="Confirmed Cases",colorscale="Reds")
fig.update(layout_coloraxis_showscale=True)
fig.show()


# In[257]:


fig = px.choropleth(s_lat_grp, locations="Country", locationmode='country names',
                    color=np.log10(s_lat_grp["Deaths"]), hover_name="Country", hover_data = ["Deaths"], 
                    color_continuous_scale="Sunsetdark", title='SARS 2003')
fig.update(layout_coloraxis_showscale=True)
fig.update_layout(title_text="Death Cases Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="Death Cases(Log Scale)",colorscale="Reds")
fig.update(layout_coloraxis_showscale=True)
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[258]:


h1n1_ds = pd.read_csv("/Users/rachanabhaskar/Desktop/paper/Data/Pandemic_H1N1_2009.csv")


# In[259]:


# ebola_ds.columns = ['Country', 'Cases', 'Deaths', 'Time']
# print(ebola_ds)


# In[260]:


h1n1_ds = h1n1_ds.groupby(['Update Time', 'Country'])['Cases', 'Deaths']
h1n1_ds = h1n1_ds.sum().reset_index()
sum(h1n1_ds['Cases'])
# print(h1n1_ds.head(100))


# In[261]:


h1n1_countries = h1n1_ds.groupby('Country')['Cases', 'Deaths'].sum().reset_index()
print(h1n1_countries)


# In[262]:


fig = px.choropleth(h1n1_countries, locations="Country", locationmode='country names',
                    color=np.log10(h1n1_countries["Cases"]), hover_name="Country", hover_data = ["Cases"], 
                    color_continuous_scale=px.colors.sequential.Plasma, title='H1N1')
fig.update(layout_coloraxis_showscale=True)
fig.update_layout(title_text="Confirmed Cases Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="Confirmed Cases(Log Scale)",colorscale="Blues")
fig.update(layout_coloraxis_showscale=True)
fig.show()


# In[263]:


fig = px.choropleth(h1n1_countries, locations="Country", locationmode='country names',
                    color=np.log10(h1n1_countries["Deaths"]), hover_name="Country", hover_data = ["Deaths"], 
                    color_continuous_scale=px.colors.sequential.Plasma, title='H1N1')
fig.update(layout_coloraxis_showscale=True)
fig.update_layout(title_text="Death Cases Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="death Cases(Log Scale)",colorscale="Reds")
fig.update(layout_coloraxis_showscale=True)
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[264]:


# MERS
mers_cntry = pd.read_csv("/Users/rachanabhaskar/Desktop/paper/Data/country_count_latest.csv")
mers_weekly = pd.read_csv("/Users/rachanabhaskar/Desktop/paper/Data/weekly_clean.csv")

# cleaning
mers_weekly['Year-Week'] = mers_weekly['Year'].astype(str) + ' - ' + mers_weekly['Week'].astype(str)
# mers_weekly['Date'] = pd.to_datetime(mers_weekly['Week'].astype(str) + 
#                                      mers_weekly['Year'].astype(str).add('-1'),format='%v%G-%u')

# mers_weekly.head()

sum(mers_cntry['Confirmed'])


# In[265]:



mers_cntry


# In[266]:


fig = px.choropleth(mers_cntry, locations="Country", locationmode='country names',
                    color=np.log10(mers_cntry["Confirmed"]), hover_name="Country", hover_data = ["Confirmed"], 
                    color_continuous_scale=px.colors.sequential.Plasma, title='MERS')
fig.update(layout_coloraxis_showscale=True)
fig.update_layout(title_text="Confirmed Cases MERS Heat Map (Log Scale)")
fig.update_coloraxes(colorbar_title="Confirmed Cases(Log Scale)",colorscale="Reds")
fig.update(layout_coloraxis_showscale=True)
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[306]:


c_cases = df_t['Confirmed'].sum()
s_cases = s_lat_grp['Cases'].sum()
e_cases = e_lat_grp['Cases'].sum()
c_deaths = df_t['Deaths'].sum()
s_deaths = s_lat_grp['Deaths'].sum()
e_deaths = e_lat_grp['Deaths'].sum()

epidemics = pd.DataFrame({
    'epidemic' : ['COVID-19', 'SARS', 'EBOLA', 'MERS', 'H1N1'],
    'start_year' : [2019, 2003, 2014, 2012, 2009],
    'end_year' : [2020, 2004, 2016, 2017, 2010],
    'confirmed' : [c_cases, s_cases, e_cases, 2494, 6724149],
    'deaths' : [c_deaths, s_deaths, e_deaths, 858, 19654],
})
# cm = sns.light_palette("green", as_cmap=True)
epidemics['mortality'] = np.round(100*epidemics['deaths']/epidemics['confirmed'],2)
epidemics = epidemics.sort_values('end_year').reset_index(drop=True)
# epidemics.style.highlight_max(color = 'lightgreen')
epidemics.style.background_gradient(cmap= 'PuBu').format("{:.0f}",subset=["confirmed"])
# epidemics.head()


# In[274]:


print(e_cases.sum())


# In[341]:


fig = px.bar(epidemics.sort_values('confirmed',ascending=False), 
             x="epidemic", y="confirmed", color='epidemic', 
             text='confirmed', title='No. of Cases', height=400, width = 500)
# fig.update_traces(textposition='auto')
# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()


# In[344]:


fig = px.bar(epidemics.sort_values('deaths',ascending=False), 
             x="epidemic", y="deaths", color='epidemic', 
             text='deaths', title='No. of deaths',height=400, width = 500)
fig.update_traces(textposition='auto')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()


# In[345]:


fig = px.bar(epidemics.sort_values('mortality',ascending=False), 
             x="epidemic", y="mortality", color='epidemic', 
             text='mortality', title=' mortality rate', height=400, width = 500)
fig.update_traces(textposition='auto')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()


# In[328]:


pos =  list(range(len(epidemics['confirmed'])))
width = 0.15
fig, ax = plt.subplots(figsize=(10,5))
plt.bar(pos, epidemics['confirmed'], width, alpha=0.5, color='#EE3224')
plt.bar([p + width for p in pos], epidemics['deaths'], width, alpha=0.5, color='#F78F1E')
plt.bar([p + width*2 for p in pos], epidemics['mortality'], width, alpha=0.5, color='#FFC222')
ax.set_xticks([p + 1.5 * width for p in pos])
ax.set_xticklabels(epidemics['epidemic'])


# In[ ]:





# In[ ]:




