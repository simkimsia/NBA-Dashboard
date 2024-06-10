from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import OrdinalEncoder
from pandas import DataFrame, concat
import streamlit as st

from interfaces.nba_stats import fetch_data_with_delays, fetch_team_data_with_delays
from NBA_helpers import clean_df

# Import Data from Session State
# region
# These are to be fetched from the session state
# could some be fetched from a global cache instead?

Team_Dict = st.session_state.Team_Dict
Teams_IDs = st.session_state.Teams_IDs
Player_IDs = st.session_state.Player_IDs
player_stats = st.session_state.player_stats
active_players = st.session_state.active_players
# endregion

if "ml_inputs" not in st.session_state:
    st.session_state.ml_inputs = []

if "X" not in st.session_state:
    st.session_state.X = DataFrame()

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
    st.session_state.Player_Dict.update(tmp_cln)
    return st.session_state.Player_Dict

def setup():
    
    st.write("ML Models Page")
    
    model = st.selectbox("Select Model", ["K-Nearest Neighbors", "Support Vector Machine"])

    nms = [x["full_name"] for x in active_players]
    obj_opts = nms + list(Teams_IDs.keys())
    st.session_state.select_obj = st.selectbox("Select Entity", obj_opts)

    return st.session_state.select_obj in nms

def is_Player():
    
    if not st.session_state.select_obj in st.session_state.Player_Dict.keys():
        st.session_state.Player_Dict = get_player_data(st.session_state.select_obj)

    if "Player" not in st.session_state or\
        st.session_state.select_obj != st.session_state["Player"]:
        st.session_state["Player"] = st.session_state.select_obj
        st.session_state['PlayerDF'] = st.session_state.Player_Dict[st.session_state.select_obj]

    def cb_rm(c):
        st.session_state.X.drop(c, axis=1, inplace=True)

    def cb_add():
        st.session_state.X[st.session_state.ML_Input] = \
            st.session_state.PlayerDF[st.session_state.ML_Input]
        st.write(st.session_state.ML_Input)

    st.session_state.ML_Input = st.selectbox("Select Stat", player_stats)

    # write a function for the below routine of checking session state & updating
            
    st.button("Add Stat", on_click=cb_add)
    
    # st.session_state.ml_inputs.append((st.write(slct), st.button("Remove Stat", on_click=cb_rm)))
    X = st.session_state.X
    st.write("Click a Stat to Remove it")

    from functools import partial
    for c in X.columns:
        st.button(c, on_click=partial(cb_rm, c))
    
    tar = st.selectbox("Add Target", st.session_state.PlayerDF.drop(X.columns, axis=1).columns)

    if not "Target" in st.session_state\
        or st.session_state.Target != tar:
        st.session_state.Target = tar
    
    st.button("Run Model")

#region

#         i = 0
#         select_stat = st.selectbox("Select Input", player_stats)
#         if not select_obj in Player_Dict.keys():
#             Player_Dict = get_player_data(select_obj)
#         df = Player_Dict[select_obj]
#         X = concat([X, df[select_stat]], axis=1)

#         with col1:
#             txt = st.write(select_stat, key=f"stat_{i}")
        
#         with col2:
#             bt = st.button("Dead Code-Remove Stat")
        
#         st.write(X.columns)

        
#         return (txt, bt)
# inputs.append(select_X())

#endregion