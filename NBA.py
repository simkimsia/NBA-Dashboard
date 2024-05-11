#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from nba_api.stats.endpoints import (commonplayerinfo, cumestatsplayer,
                                     playergamelog)
from nba_api.stats.static import players

from NBA_helpers import clean_df, map_wl

# read player names by command line
lst = ["Jamal Murray", "Nikola Jokic", "Kentavious Caldwell", "Paul George", "Mason Plumlee"]
lg = "00"
plyrs = players.get_players()

# Creating dictionary of active player names and their IDs
plyr_names = map(lambda y: y["full_name"], filter(lambda x: x['is_active'] == True, plyrs))
plyr_names = list(plyr_names)
r = {nm : players.find_players_by_full_name(nm)[0]["id"] for nm in lst}
dat = {nm : playergamelog.PlayerGameLog(player_id=r[nm], season="2023").get_data_frames()[0] for nm in r.keys()}

# Data cleaning and preparation
d = dict(zip(dat.keys(), list(map(clean_df, dat.values()))))

stats = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
         'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']

# Streamlit layout
st.title("Player Viz")
selected_stat = st.selectbox("Select Stat", stats)
selected_player = st.selectbox("Select Player", lst)

# Define a function to plot data
def plot_data(player, stat):
    df = d[player]
    fig = px.scatter(df,
                    x="GameDate",
                    y=stat,
                    color="WL",
                    color_discrete_sequence=["red", "green"],
                    title=f"{player} {stat} by Game")
    return fig

# Plotting
fig = plot_data(selected_player, selected_stat)
st.plotly_chart(fig)
