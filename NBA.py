#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo, cumestatsplayer
from nba_api.stats.endpoints import playergamelog
from NBA_helpers import  map_wl, clean_df

from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px


# In[ ]:


# read player names by command line
lst = ["Jamal Murray", "Nikola Jokic", "Kentavious Caldwell", "Paul George", "Mason Plumlee"]
lg = "00"
plyrs = players.get_players()


# # Player Search
# * Can find `player_id` by Full Name
# * DataBase of Player names can be quirky, maybe a regex?
# * TODO:: Review Docstring for method to determine unexpected behavior (0 matches/multiple matches)

# In[ ]:


plyr_names = map(lambda y: y["full_name"], filter(lambda x: x['is_active'] == True, plyrs))
plyr_names = list(plyr_names)
r = {nm : players.find_players_by_full_name(nm)[0]["id"] for nm in lst}
dat = {nm : playergamelog.PlayerGameLog(player_id=r[nm], season="2023").get_data_frames()[0] for nm in r.keys()}


# Each item in this dictionary is a `{name : pandas DataFrame}` pair. The DataFrame `info()` output is below

# # DataFrame Description/Cleaning
# * Current
#     * `WL` column is "L" or "W", object dtype. Map to 1:W, 0:L
#     * `Matchup` contains opponent detail as well as home/away. Should attempt two RegEx matches in order to determine Home/Away status and then strip the opponent data. Return both pieces of info as a tuple, add each item in tuple to DataFrame. Once a stable method is in place, `git commit` then see how ChatGPT does
#     * `GAME_DATE` might convert right into a DateTime object
#     * `SEASON_ID` and `Game_ID` are nominal-numeric, unsure of what they could be used for... consult docs? TODO:: find docs
# 
# 
# * Future
#     * Video Available seems cool, will have to call `help` on library function that retrieves DataFrame

# In[ ]:


# Create a dictionary of player names : statistics
d = dict(
            zip(dat.keys(), list( map(clean_df, dat.values()) ))
        )


# In[ ]:


stats = ['MIN', 'FGM',
       'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
       'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
       'PLUS_MINUS']


# In[ ]:


df = d["Paul George"]

app = Dash("Player Viz")

app.layout = html.Div([
    html.Div([dcc.Dropdown(id="stat", options=stats, value="PTS")], id="stat_controls"),
    html.Div([dcc.Dropdown(id="player", options=[{"label": nm, "value": nm} for nm in lst], value=lst[0])], id="plyr_controls"),
    html.Button("Add Stat", id="stat_add"),
    html.Button("Remove Stat", id="stat_remove"),
    # html.Button("Add Player", id="player_add"),
    # html.Button("Remove Player", id="player_remove"),
    dcc.Graph(figure = {}, id="controls-and-graph")
])

@callback(
        Output("stat_controls", "children"),
        Input("stat_add", "n_clicks"),
        Input("stat_remove", "n_clicks"),
        State("stat_controls", "children"),
        allow_duplicate=True
)
def add_stats(y_clicks, n_clicks, curr_children):
    if y_clicks:
        return dcc.Dropdown(id="stat", options=stats, value=stats[0])
    elif n_clicks:
        if curr_children:
            curr_children.pop()
            return curr_children
        else:
            return []

    
@callback(
    Output(component_id="controls-and-graph", component_property="figure"),
    Input(component_id="controls-and-graph", component_property="figure"),
    Input(component_id="player", component_property="value"),
    Input(component_id="stat", component_property="value")
)
def update_graph(fig, player, stat):
    if fig:
        fig['data'].append(px.line(x=d[player].GameDate, y=d[player][stat], title=f"{player} {stat} by Game"))
    else:
        df = d[player]
        fig = px.scatter(df, 
                        x="GameDate",
                        y=stat, 
                        color="WL", 
                        color_discrete_sequence=["red", "green"],
                        #  trendline="mean",
                        #  color_discrete_map={0:"red", 1:"green"}, 
                        title=f"{player} {stat} by Game"
                        )
    return fig


# # Run all above to compile dashboard, below will execute/display it

# In[ ]:


app.run(debug=True)

