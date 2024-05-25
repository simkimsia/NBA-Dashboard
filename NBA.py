#!/usr/bin/env python
# coding: utf-8

import plotly.express as px
import streamlit as st
from nba_api.stats.static import players

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

# Example team IDs, replace with actual values
team_ids = {"Lakers": "1610612747", "Nuggets": "1610612743"}

# Fetch game log data for teams with delays
team_data = fetch_team_data_with_delays(team_ids)

st.title("NBA Visualization")

# Toggle between Player and Team Stats
view_mode = st.radio("View Mode", ["Player Stats", "Team Stats"])


if view_mode == "Player Stats":
    # Fetch player IDs dynamically when Player Stats is selected
    plyrs = players.get_players()
    r = {
        nm: players.find_players_by_full_name(nm)[0]["id"]
        for nm in player_names
        if players.find_players_by_full_name(nm)
    }

    # Fetch and clean player data
    player_data = fetch_data_with_delays(r)
    player_data_cleaned = {key: clean_df(val) for key, val in player_data.items()}

    selected_stat = st.selectbox("Select Stat", player_stats)
    selected_player = st.selectbox("Select Player", player_names)

    def plot_data(player, stat):
        df = player_data_cleaned[player]
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
    team_data = fetch_team_data_with_delays(team_ids)
    team_data_cleaned = {key: clean_df(val) for key, val in team_data.items()}

    selected_stat = st.selectbox("Select Team Stat", team_stats)
    selected_team = st.selectbox("Select Team", list(team_ids.keys()))

    def plot_team_data(team, stat):
        df = team_data_cleaned[team]
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
