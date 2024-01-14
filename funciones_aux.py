# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:49:50 2024

@author: alexb
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, shape
from shapely import ops
import pyproj
import plotly.express as px
from streamlit_folium import st_folium
import plotly.express as px
import streamlit as st 

###Cargado de datos########################################################################################################################################################
#Cargamos los dos tipos de fichero que necesitamos para la practica
def cargar_ficheros(fichero):
    if 'zip' in fichero:
        shp_path = fichero

        # Carga el archivo SHP en un GeoDataFrame
        gdf = gpd.read_file(shp_path)
        return gdf
    else:
        with open(fichero, 'r') as file:
            json_data = file.read()
        # Cargar el JSON en un diccionario
        data_dict = json.loads(json_data)
        
        # Obtener la lista 'data' del diccionario
        data_list = data_dict.get('data', [])
        
        # Crear un DataFrame de Pandas con la lista 'data'
        df = pd.DataFrame(data_list)
        return df

###Preparacion de los datos###################################################################################################################################################

#Modificamos la estructura del df para poder representar los datos en un mapa, 
#la opcion añadir_geometry_inverted se incluye para el caso en el que queremos relacionar el df con gdf
def modificar_col_geometry(df,añadir_geometry_inverted=False):
    if añadir_geometry_inverted:
        df['geometry_inverted'] = df['geometry'].apply(lambda x: Point(x['coordinates'][0],x['coordinates'][1]))
        df['geometry'] = df['geometry'].apply(lambda x: Point(x['coordinates'][1],x['coordinates'][0]))
        df['distrito']=None
    else:
        df['latitud'] = df['geometry'].apply(lambda x: x['coordinates'][1])
        df['longitud'] = df['geometry'].apply(lambda x: x['coordinates'][0])
        df['geometry'] = df['geometry'].apply(lambda x: x['type'])
    return df

#Esta funcion se implementa para cambiar la proyeccion del poligono de forma que se pueda relacionar con puntos del mapa 
def cambio_proyeccion(poly):
    poly_crs = pyproj.CRS("EPSG:25830")
    poly = ops.transform(lambda x, y: pyproj.Transformer.from_crs(poly_crs, pyproj.CRS("EPSG:4326"), always_xy=True).transform(x,y), poly)
    return poly


#Esta funcion sirve para crear un df que contenga informacion sobre que distrito corresponde a cada estacion
def estaciones_con_distrito(df,gdf):
    df=modificar_col_geometry(df,True)   
    gdf=gdf.assign(changes=gdf['geometry'])
    gdf['changes']=gdf['changes'].apply(cambio_proyeccion)    
    
    def identificar_distrito(estacion):
        al_reves = gdf['changes'].contains(estacion)
        esta = gdf['NOMBRE'].loc[al_reves]
        return esta.iloc[0] 

    df['distrito']=df['geometry_inverted'].apply(identificar_distrito)
    gdf=gdf.iloc[:, :-1]
    df_con_distrito=df
    return df_con_distrito  

#Se agrupa por distrito el numero de estaciones
def num_estaciones_por_distrito(df_con_distrito):
    df_estaciones_por_distrito=df_con_distrito['distrito'].value_counts().reset_index()
    return df_estaciones_por_distrito

#Se añade informacion en gdf sobre la cantidad de estaciones que hay en cada distrito
def modificar_gdf(gdf,df_estaciones_por_distrito):
    gdf_con_estaciones=gdf.merge(df_estaciones_por_distrito,left_on='NOMBRE', right_on='distrito',how='left')
    gdf_con_estaciones['COD_DIS_TX']=gdf_con_estaciones['count']
    gdf_con_estaciones=gdf_con_estaciones.iloc[:, :-2]
    gdf_con_estaciones = gdf_con_estaciones.rename(columns={'COD_DIS_TX': 'Estaciones'})
    return gdf_con_estaciones

#Se realiza un filtrado de las bases inferiores a un numero dado
def cantidad_estaciones(df,num):
    df_cantidad=df.loc[df['total_bases']>=num]
    return df_cantidad



###Display################################################################################################################################################################

#Se representan de manera simple todas las estaciones en un mapa
def mostrar_estaciones_bicimad_de_golpe(df):
    def agregar_marcador(fila):
        folium.Marker(location=[fila['latitud'], fila['longitud']], popup=fila['name']).add_to(map)
    map=folium.Map(location=[df['latitud'].mean(),df['longitud'].mean()], zoom_start=11, scrollWheelZoom=False, tiles='CartoDB positron')
    # Aplicar la función a cada fila del DataFrame
    df.apply(agregar_marcador, axis=1)
    st_map=st_folium(map, width=700, height=450)    

#Se emplea la funcion cluster para representar de manera más cómoda todas las estaciones en el mapa
def mostrar_estaciones_bicimad_cluster(df):
    def agregar_marcador1(fila):
        folium.Marker(location=[fila['latitud'], fila['longitud']], popup=fila['name']).add_to(marker_cluster)

    map=folium.Map(location=[df['latitud'].mean(),df['longitud'].mean()], zoom_start=11, scrollWheelZoom=False, tiles='CartoDB positron')
    marker_cluster = MarkerCluster().add_to(map)
    # Aplicar la función a cada fila del DataFrame
    df.apply(agregar_marcador1, axis=1)
    st_map=st_folium(map, width=700, height=450)    
    

#Se representa en el mapa los distritos de madrid
def mostrar_distritros(gdf):
    def style_function(feature):
        return {
            'fillColor': 'blue',  # Color de relleno
            'color': 'black',      # Color de borde
            'weight': 1,           # Grosor de la línea
            'opacity': 0.5,         # Opacidad de la línea
            'fillOpacity': 0.3
        }
    map=folium.Map(location=[40.416767, -3.681854], zoom_start=10, scrollWheelZoom=False, tiles='CartoDB positron')
    folium.GeoJson(gdf,style_function=style_function,
                       popup = folium.GeoJsonPopup(fields = ['NOMBRE'],
                                aliases=['Distrito: '],
                                localize=True,
                                labels=True,
                                parse_html=False)).add_to(map)
    st_map=st_folium(map, width=700, height=450)

#Funcion similar a mostrar_distritros que añade una capa en la que se representan en cluster las estaciones de bici
def mostrar_distritros_con_estaciones(gdf,df):
    def agregar_marcador1(fila):
        folium.Marker(location=[fila['latitud'], fila['longitud']], popup=fila['name']).add_to(marker_cluster)

    map=folium.Map(location=[df['latitud'].mean(),df['longitud'].mean()], zoom_start=11, scrollWheelZoom=False, tiles='CartoDB positron')
    marker_cluster = MarkerCluster().add_to(map)
    # Aplicar la función a cada fila del DataFrame
    df.apply(agregar_marcador1, axis=1)
    folium.GeoJson(gdf,popup = folium.GeoJsonPopup(fields = ['NOMBRE'],
                                aliases=['Distrito: '],
                                localize=True,
                                labels=True,
                                parse_html=False)).add_to(map)
    st_map=st_folium(map, width=700, height=450)    
    

#Representa por escala de colores en el mapa la cantidad de estaciones por distrito
def mostrar_mapa_cloropleth(df_estaciones_por_distrito,gdf_con_estaciones):
    map=folium.Map(location=[40.42514422131318, -3.6833399438991723], zoom_start=10, scrollWheelZoom=False, tiles='CartoDB positron')
    folium.Choropleth(
        geo_data=gdf_con_estaciones,
        name='choropleth',
        data = df_estaciones_por_distrito,
        columns=['distrito', 'count'],
        key_on='feature.properties.NOMBRE',
        fill_color='YlGn',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Numero de estaciones de bici'
    ).add_to(map)
    
    def style_function(feature):
        return {
            'fillColor': 'green',  # Color de relleno
            'color': 'black',      # Color de borde
            'weight': 1,           # Grosor de la línea
            'opacity': 0.5,         # Opacidad de la línea
            'fillOpacity': 0.0
        }    
    folium.GeoJson(gdf_con_estaciones,
                   style_function=style_function,
                   popup = folium.GeoJsonPopup(fields = ['NOMBRE','Estaciones'],
                                               aliases=['Distrito:','Estaciones: '],
                                               localize=True,
                                               labels=True,
                                               parse_html=False
                                                        )).add_to(map)
    st_map=st_folium(map, width=700, height=450)
    
    
    
def mostrar_mapa_densidad(df):
    fig = px.density_mapbox(df, lat = 'latitud', lon = 'longitud', z = 'total_bases',
                            radius = 10,
                            center = dict(lat = df['latitud'].mean(), lon = df['longitud'].mean()),
                            zoom = 10,
                            mapbox_style = 'open-street-map')    
    return fig


def mostrar_grafico(df,gdf,percent):
    df_copia_grafico = df.copy(deep=True)
    gdf_copia_grafico = gdf.copy(deep=True)
    df_con_distrito,df_estaciones_por_distrito,_=df_con_distrito_y_gdf_con_estaciones(df_copia_grafico,gdf_copia_grafico)
    total_bicis=df_con_distrito.groupby('distrito')['total_bases'].sum().reset_index()
    total_bicis_estacion=total_bicis.merge(df_estaciones_por_distrito,on='distrito',how='left')
    columna=None
    if percent=='ratio':
        total_bicis_estacion=total_bicis_estacion.assign(changes=total_bicis_estacion['total_bases']/total_bicis_estacion['count'])
        columna='changes'
    else:
        columna='total_bases'
    total_bicis_estacion=total_bicis_estacion.sort_values(by=columna)
    fig = px.bar(total_bicis_estacion, x='distrito', y=columna, 
             labels={'distrito': 'Distrito', columna: 'Bicis'})

    # Puedes ajustar el ángulo del texto del eje x
    fig.update_xaxes(tickangle=45, tickmode='array')
    
    return fig
###Funciones principales########################################################################################################################################


def cargar_datos():
    df=cargar_ficheros('response.json')
    gdf=cargar_ficheros('Distritos.zip')
    return df,gdf

def botones_para_modificar_df(df, distritos=False):
    if distritos=='Solo distritos':
        return df
    else:
        opciones=['Activas','Inactivas']
        seleccion=st.multiselect('Estado de la estacion:', opciones)
        if len(seleccion)==1:
            if 'Activas' in seleccion:
                seleccion_1=0
            else:
                seleccion_1=1
                
            df=df.loc[df['no_available']==seleccion_1]
        min_df=df['total_bases'].loc[(df['total_bases']==min(df['total_bases']))].iloc[0]
        max_df=df['total_bases'].loc[(df['total_bases']==max(df['total_bases']))].iloc[0]
        num = st.slider('Número mínimo de bicis por estación',min_df,max_df)
        df=cantidad_estaciones(df,num)
        return df


def estaciones_option(df):
    col1, col2 = st.columns([1, 3])
    with col1:
      st.header("Opciones")
      sidebar_estaciones = st.selectbox("Tipo de mapa de las estaciones:",
           ("Simple", "Cluster")
       )
      df=botones_para_modificar_df(df)
      
    with col2:
      df=modificar_col_geometry(df)
      if sidebar_estaciones=='Simple':
          st.header("Mapa de las estaciones simple") 
          mostrar_estaciones_bicimad_de_golpe(df)
      if sidebar_estaciones=='Cluster':
          st.header("Mapa de estaciones cluster")
          mostrar_estaciones_bicimad_cluster(df)

def distrito_option(gdf,df):
    col1, col2 = st.columns([1, 3])
    with col1:
      st.header("Opciones")
      sidebar_distritos= st.selectbox('Tipo de mapa de las estaciones:',
           ("Solo distritos", "Distritos con estaciones"))
      df=botones_para_modificar_df(df,sidebar_distritos)
      df=modificar_col_geometry(df)
      
    with col2:
        if sidebar_distritos=='Solo distritos':
            st.header("Solo distritos")
            mostrar_distritros(gdf)
        if sidebar_distritos=='Distritos con estaciones':
            st.header("Distritos con estaciones")
            mostrar_distritros_con_estaciones(gdf,df)

def df_con_distrito_y_gdf_con_estaciones(df,gdf):
    df_con_distrito=estaciones_con_distrito(df,gdf)
    df_estaciones_por_distrito=num_estaciones_por_distrito(df_con_distrito)
    gdf_con_estaciones=modificar_gdf(gdf,df_estaciones_por_distrito)
    return df_con_distrito,df_estaciones_por_distrito,gdf_con_estaciones
        
        
def analisis_option(df,gdf):
    col1, col2 = st.columns([1, 3])
    with col1:
      st.header("Opciones")
      sidebar_distritos_estacion= st.selectbox('Tipo de mapa de las estaciones:',
           ("Mapa con escala de color", "Mapa de densidad"))
      df=botones_para_modificar_df(df)
      
    with col2:
        if sidebar_distritos_estacion=='Mapa de densidad':
            st.header("Mapa de densidad")
            df_copia_profunda = df.copy(deep=True)
            df_adecuado=modificar_col_geometry(df_copia_profunda)
            st.plotly_chart(mostrar_mapa_densidad(df_adecuado))
        
        if sidebar_distritos_estacion=='Mapa con escala de color':
           st.header("Mapa con escala de color")
           df_copia = df.copy(deep=True)
           gdf_copia = gdf.copy(deep=True)
           _,df_estaciones_por_distrito,gdf_con_estaciones=df_con_distrito_y_gdf_con_estaciones(df_copia,gdf_copia)
           mostrar_mapa_cloropleth(df_estaciones_por_distrito,gdf_con_estaciones)
    

def menu():
    #Indicamos seleccion
    opcion_principal=st.sidebar.radio('¿Qué ver?',('Mapas', 'Grafico'))
    #Cargamos los datos  
    df,gdf=cargar_datos()
    
    #Botones para filtrar
    #df=botones_para_modificar_df(df)
    df_copia1 = df.copy(deep=True)
    gdf_copia1 = gdf.copy(deep=True)
    if opcion_principal=='Mapas':
        menu=st.sidebar.selectbox("Elige mapa un tipo de mapa",
             ("Estaciones", "Distritos","Analisis"))
        if menu=="Estaciones":
           estaciones_option(df) 
        if menu=="Distritos":
           distrito_option(gdf,df)
        if menu=="Analisis":
           analisis_option(df,gdf)
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
          df=botones_para_modificar_df(df)
          opcion_grafico = st.radio(
               "Elige como ver el grafico",
          ('bicis totales', 'ratio'))
          
        with col2:   
            st.plotly_chart(mostrar_grafico(df,gdf_copia1,opcion_grafico))



























