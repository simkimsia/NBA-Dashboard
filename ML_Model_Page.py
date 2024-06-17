from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import OrdinalEncoder
from pandas import DataFrame, concat
import streamlit as st
from NBA_helpers import player_stats, team_stats
from interfaces.nba_stats import fetch_data_with_delays, fetch_team_data_with_delays
from NBA_helpers import clean_df
from matplotlib import pyplot as plt

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

def cb_rm(c):
    st.session_state.X.drop(c, axis=1, inplace=True)

def cb_add():
    st.session_state.X[st.session_state.ML_Input] = \
        st.session_state.PlayerDF[st.session_state.ML_Input]
    st.write(st.session_state.ML_Input)

def setup():
    
    st.write("ML Models Page")
    
    st.session_state.model = st.selectbox("Select Model", ["K-Nearest Neighbors", "Support Vector Machine"])

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

    st.session_state.ML_Input = st.selectbox("Select Stat", player_stats)

    # write a function for the below routine of checking session state & updating
            
    st.button("Add Stat", on_click=cb_add)
    
    # st.session_state.ml_inputs.append((st.write(slct), st.button("Remove Stat", on_click=cb_rm)))
    X = st.session_state.X
    st.write("Click a Stat to Remove it")

    from functools import partial
    for c in X.columns:
        st.button(c, on_click=partial(cb_rm, c))
    

    if not "Target" in st.session_state:
        st.session_state.Target = st.selectbox("Add Target", st.session_state.PlayerDF.drop(X.columns, axis=1).columns)
    else:
        ind = st.session_state.PlayerDF.drop(X.columns, axis=1).columns.get_loc(st.session_state.Target)
        st.write(ind)
        st.session_state.Target = st.selectbox("Change Target", 
                                               st.session_state.PlayerDF.drop(X.columns, axis=1).columns,
                                               index=ind)

    def run_model():
        m = st.session_state.model
        X = st.session_state.X
        tname = st.session_state.Target
        tar = st.session_state.PlayerDF[tname]
        cols = X.columns
        shp = X.shape
        # X = X.to_numpy().reshape(-1, 1)
        X = X.to_numpy()
        tar = tar.to_numpy()
        if m == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
        elif m == "Support Vector Machine":
            model = SVC()
        model.fit(X, tar)
        fig, ax = plt.subplots()
        # TODO:: adjust labels for more than 2 features
        DecisionBoundaryDisplay.from_estimator(
            model, 
            X, 
            response_method="predict", 
            ax=ax,
            xlabel=cols[0],
            ylabel=cols[1]
        )

        if shp[1] > 2:
            st.write("More than 2 features, plot has been adjusted")
            if shp[1] == 3:
                plt.scatter(X[:, 0], X[:, 1], c=X[:, 2])
                plt.legend(cols[2])
                st.write(f"Coloring by {X.columns[2]}")
            elif shp[1] == 4:
                plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], s=X[:, 3])
                plt.legend(cols[2:3])
                st.write(f"Coloring by {X.columns[2]} and sizing by {X.columns[3]}")
            else:
                st.write("Too many features to plot")
        plt.title(f"{m} Decision Boundary\nTarget {tname}\nFeatures {' '.join(cols.tolist())}")
        plt.xlabel = cols[0]
        plt.ylabel = cols[1]
        # plt.colorbar()
        st.pyplot(fig)

    st.button("Run Model", on_click=run_model)

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