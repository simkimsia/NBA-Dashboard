#!/usr/bin/env python
# coding: utf-8

import plotly.express as px
import streamlit as st
from nba_api.stats.static import players, teams
from pandas import DataFrame

from interfaces.nba_stats import fetch_data_with_delays, fetch_team_data_with_delays
from NBA_helpers import clean_df

# Example team and player setup
team_ids = {"Lakers": "1610612747", "Nuggets": "1610612743"}
player_names = [
    "Jamal Murray",
    "Nikola Jokic",
    "Kentavious Caldwell",
    "Paul George",
    "Mason Plumlee",
]

# Define stats options
player_stats = [
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "PLUS_MINUS",
]
team_stats = [
    "W",
    "L",
    "W_PCT",
    "PTS",
    "REB",
    "AST",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "STL",
    "BLK",
    "TOV",
    "PF",
]

team_data = teams.get_teams()
Teams_IDs = {team["full_name"]: team["id"] for team in team_data}
Team_Dict = {}
# team_data = fetch_team_data_with_delays(team_ids)
# team_data_cleaned = {key: clean_df(val) for key, val in team_data.items()}


player_data = players.get_players()
active_players = [player for player in player_data if player["is_active"]]
Player_IDs = {player["full_name"]: player["id"] for player in active_players}
Player_Dict = {}
# player_data_cleaned = {key: clean_df(val) for key, val in player_data.items()}

@st.cache_data
def get_team_data(team_name):
    tmp = fetch_team_data_with_delays({team_name: Teams_IDs[team_name]})
    tmp_cln = {team_name : clean_df(tmp[team_name])}
    Team_Dict.update(tmp_cln)
    return Team_Dict

@st.cache_data
def get_player_data(player_name):
    tmp = fetch_data_with_delays({player_name: Player_IDs[player_name]})
    tmp_cln = {player_name : clean_df(tmp[player_name])}
    Player_Dict.update(tmp_cln)
    return Player_Dict

# get_player_data("Jamal Murray")


# # Example team IDs, replace with actual values
# team_ids = {"Lakers": "1610612747", "Nuggets": "1610612743"}

# Fetch game log data for teams with delays
team_data = fetch_team_data_with_delays(Teams_IDs)

st.title("NBA Visualization")

# Toggle between Player and Team Stats
view_mode = st.radio("View Mode", ["Player Stats", "Team Stats"])


    # Lets not fetch data dynamically, the only advantage that will provide is for
    # streaming; that should be a premium feauture of our end product

    # Also, I think this current view_mode branching makes the code a bit long unnecessarily

if view_mode == "Player Stats":

    selected_stat = st.selectbox("Select Stat", player_stats)
    selected_player = st.selectbox("Select Player", player_names)

    if not selected_player in Player_Dict.keys():
        Player_Dict = get_player_data(selected_player)

    def plot_data(player, stat):
        df = Player_Dict[player]
        if df.empty:
            st.error(f"No data available for {player}. Please try another player.")
            return None
        fig = px.scatter(
            df,
            x="GameDate",
            y=stat,
            color="WL",
            color_discrete_sequence=["red", "green"],
            title=f"{player} {stat} by Game",
        )
        return fig

    fig = plot_data(selected_player, selected_stat)
    st.plotly_chart(fig)

else:
    # Only fetch team data if the Team Stats view is selected

    selected_stat = st.selectbox("Select Team Stat", team_stats)
    selected_team = st.selectbox("Select Team", list(Teams_IDs.keys()))

    if selected_team not in Team_Dict.keys():
        Team_Dict = get_team_data(selected_team)

    def plot_team_data(team, stat):
        df = Team_Dict[team]
        if df.empty:
            st.error(f"No data available for {team}. Please try another team.")
            return None
        # Ensure the column exists in DataFrame
        if stat not in df.columns:
            st.error(f"Stat {stat} not available. Please choose another stat.")
            return None
        fig = px.line(
            df,
            x="GAME_DATE",  # Make sure this column exists and is in the correct format
            y=stat,  # This will now reference an existing column correctly
            title=f"{team} {stat} by Game",
        )
        return fig

    fig = plot_team_data(selected_team, selected_stat)
    st.plotly_chart(fig)
