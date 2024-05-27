#!/usr/bin/env python
# coding: utf-8

import plotly.express as px
import streamlit as st
from nba_api.stats.static import players, teams
from pandas import DataFrame, concat
# from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import OrdinalEncoder
import datetime

from interfaces.nba_stats import fetch_data_with_delays, fetch_team_data_with_delays
from NBA_helpers import clean_df

# dummy data below
#region
# Example team and player setup
# team_ids = {"Lakers": "1610612747", "Nuggets": "1610612743"}
# player_names = [
#     "Jamal Murray",
#     "Nikola Jokic",
#     "Kentavious Caldwell",
#     "Paul George",
#     "Mason Plumlee",
# ]
#endregion


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


player_data = players.get_players()
active_players = [player for player in player_data if player["is_active"]]
Player_IDs = {player["full_name"]: player["id"] for player in active_players}
Player_Dict = {}

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

# Fetch game log data for teams with delays
team_data = fetch_team_data_with_delays(Teams_IDs)

st.title("NBA Visualization")

def date_slider(dates):
    min_date = dates.min().to_pydatetime()
    max_date = dates.max().to_pydatetime()
    step = datetime.timedelta(days=1)
    return st.slider("Select Date Range", value=(min_date,max_date), step=step, format='MMM DD, YYYY')

page_select = st.selectbox("Select Page", ["Simple Graphs", "ML Models",
                                          "Simulations"])

if page_select == "Simple Graphs":
    st.write("Simple Graphs Page")

    # Toggle between Player and Team Stats
    view_mode = st.radio("View Mode", ["Player Stats", "Team Stats"])

    # ML methods should be made available inside of an ML Specific view
    # region
    # """ 
    # knn = st.checkbox("K-Nearest Neighbors Clustering")
    # svc = st.checkbox("Support Vector Machine")

    # kmn = st.checkbox("K-Means Clustering") """

        # Lets not fetch data dynamically, the only advantage that will provide is for
        # streaming; that should be a premium feauture of our end product

        # Also, I think this current view_mode branching makes the code a bit long unnecessarily
    # endregion

    def view_team_stats(Team_Dict=Team_Dict):
        selected_stat = st.selectbox("Select Team Stat", team_stats)
        selected_team = st.selectbox("Select Team", list(Teams_IDs.keys()))

        if selected_team not in Team_Dict.keys():
            Team_Dict = get_team_data(selected_team)

        def plot_team_data(team, stat):
            df = Team_Dict[team]
            if df.empty:
                st.error(f"No data available for {team}. Please try another team.")
                return None
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
        return (selected_team, selected_stat, plot_team_data)


    def view_player_stats(Player_Dict=Player_Dict):
        selected_stat = st.selectbox("Select Stat", player_stats)
        selected_player = st.selectbox("Select Player", list(Player_IDs.keys()))
        
        if not selected_player in Player_Dict.keys():
            Player_Dict = get_player_data(selected_player)
        Player_df = Player_Dict[selected_player]
        slider = date_slider(Player_df["GameDate"])

        def plot_data(player, stat):
            df = Player_Dict[player]
            df = df[(df["GameDate"] >= slider[0]) & (df["GameDate"] <= slider[1])]
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
        
        return (selected_player, selected_stat, plot_data)
                            
    if view_mode == "Player Stats":
        selection, selected_stat, plot_data = view_player_stats()
    
    elif view_mode == "Team Stats":
        selection, selected_stat, plot_data = view_team_stats()

    fig = plot_data(selection, selected_stat)
    st.plotly_chart(fig)

elif page_select == "ML Models":
    st.write("ML Models Page")

elif page_select == "Simulations":
    st.write("Simulations Page")

# region
# else:

    # Only fetch team data if the Team Stats view is selected

    # fig = plot_team_data(selected_team, selected_stat)
    # st.plotly_chart(fig)


    # Below is dead code for KNN and SVC, I think we should have a separate view for ML methods
# """ 
#         if knn:
#             # Fit a KNN model to the data
#             e = OrdinalEncoder()
#             dt = e.fit_transform(df['GameDate'].to_numpy().reshape(1, -1))
#             m = KNeighborsClassifier(n_neighbors=2)
#             tar = df["WL"].apply(lambda x: 1 if "W" else 0).to_numpy().reshape(-1, 1)
#             m.fit(concat([dt, df[stat]], axis=1), tar)
#             y_pred = m.predict(tar)
#             df["Predicted"] = y_pred
#             DecisionBoundaryDisplay.from_estimator(m, 
#                                                    df[[stat, dt]],
#                                                    response_method="predict",
#                                                    ax=fig
#                                                   )

#         # SVC is going to work better as it's own view, so you can add >2 params to fit.
        
#         # Would multiclassification add value?

#         elif svc:
#             # Fit a Support Vector Machine model to the data
#             m = SVC(kernel="linear")
#             tar = df["WL"].apply(lambda x: 1 if "W" else 0).to_numpy().reshape(-1, 1)
#             m.fit(df[[stat]], tar)
#             y_pred = m.predict(tar)
#             df["Predicted"] = y_pred
            
#             DecisionBoundaryDisplay.from_estimator(m, 
#                                                     df[["GameDate", stat]],
#                                                     response_method="predict",
#                                                     ax=fig)
#                     """       
# endregion