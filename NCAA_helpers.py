from scipy import stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import pickle
pd.options.mode.copy_on_write = True

team_df = pd.read_csv('mar25data/NCAAteams.csv')
plyr_df = pd.read_csv('mar25data/NCAAplyrs.csv')

with open("teamstats.pkl", "rb") as f:
    tm_stats = pickle.load(f)

def update_data():
    team_df = pd.read_csv('NCAAteams.csv')
    plyr_df = pd.read_csv('NCAAplyrs.csv')

def read_alt(path):
    team_df = pd.read_csv(path + '/NCAAteams.csv')
    plyr_df = pd.read_csv(path + '/NCAAplyrs.csv')

teams = pd.concat([team_df.home_team,team_df.away_team], axis=0).unique()

# Helper method for PtAdds
def addOpp(x, team):
    id = x.game_id
    gm = team_df[team_df.game_id == id]
    if gm.home_team.values[0] == team:
        return gm.away_team.values[0]
    else:
        return gm.home_team.values[0]

# Todo :: add functionality and rename
# Add Home/Away, PtsFor, PtsAgainst, TotalPts, Opp columns to DataFrame
def PtAdds(df, nm):
    df["Home/Away"] = np.where(df.home_team == nm, "Home", "Away")
    df["PtsFor"] = np.where(df.home_team == nm, df.home_score, df.away_score)
    df["PtsAgainst"] = np.where(df.home_team == nm, df.away_score, df.home_score)
    df["TotalPts"] = df.PtsFor + df.PtsAgainst
    df["Opp"] = df.apply(addOpp, axis=1, args=(nm,))
    return df

def tm_descriptive(team):
    tm = team_df[np.logical_or(team_df.home_team == team, team_df.away_team == team)]
    tm = PtAdds(tm, team)
    tm["Spread"] = tm.PtsFor - tm.PtsAgainst
    tm["Win"] = np.where(tm.PtsFor > tm.PtsAgainst, True, False)
    d = {"AvgPtsFor":tm.PtsFor.mean(), "StdPtsFor":tm.PtsFor.std(),
         "AvgPtsAgainst":tm.PtsAgainst.mean(), "StdPtsAgainst":tm.PtsAgainst.std(),
         "AvgTotalPts":tm.TotalPts.mean(), "StdTotalPts":tm.TotalPts.std(),
         "WinPct":tm.Win.mean(), 
         "AvgWinSpread" : tm[tm.Win].Spread.mean(), "StdWinSpread":tm[tm.Win].Spread.std(),
         "AvgLossSpread":tm[np.logical_not(tm.Win)].Spread.mean(), "StdLossSpread":tm[np.logical_not(tm.Win)].Spread.std(),
         "AvgSpread":tm.Spread.mean(), "StdSpread":tm.Spread.std()}
    tm_stats[team] = d
    return d

# Search teams for substring s. DataFrame is default Players DataFrame
def tm_search(s, df=plyr_df):
    lst = np.array([x for x in teams if s in x])
    return lst

def get_team_games(team):
    tm = team_df[np.logical_or(team_df.home_team == team, team_df.away_team == team)]
    plyr = plyr_df[plyr_df.team == team]
    tm = PtAdds(tm, team)
    tm["Spread"] = tm.PtsFor - tm.PtsAgainst
    tm["Win"] = np.where(tm.PtsFor > tm.PtsAgainst, True, False)
    return {"team":tm, "players":plyr}

# Return True if col in df is greater than prop, False otherwise
def prop_hit(col, prop, df):
    return np.where(df[col] > prop, True, False)

# Return the scalar distance from prop for col in df
def prop_build(col, prop, df):
    # copilot returned bad code, and I didn't check it
    # it was a fucking one liner... whatever fixed it
    return df[col] - prop

# Return Percentage of Games where prop hits for col in df
def prop_pct(col, prop, df):
    return np.sum(prop_hit(col, prop, df))/len(df)
# make a method to replace the below cell of getting all the teams for a given days games
# parameter should be a list of tuples [(tm1, tm2),,,]
def create_game_data(team_tuples):
    game_data = []
    
    for tm1, tm2 in team_tuples:
        game_dict = {}
        nm1 = tm_search(tm1)
        nm2 = tm_search(tm2)
        fail = False
        if len(nm1) > 1:
            print(f"Non-unique id {tm1}, returning {nm1[0]}")
        if len(nm2) > 1:
            print(f"Non-unique id {tm2}, returning {nm2[0]}")
            
        if len(nm1) == 0:
            print(f"No team found for {tm1}")
            fail = True
        if len(nm2) == 0:
            print(f"No team found for {tm2}")
            fail = True
        if fail:
            continue
            
        tm1 = nm1[0]
        tm2 = nm2[0]

        game_dict[tm1] = get_team_games(tm1)
        game_dict[tm2] = get_team_games(tm2)
        
        game_data.append(game_dict)

    return game_data

def regression(df, x, y):
    if not x in df.columns or not y in df.columns:
        msg = "Neither Column in DataFrame" if not x in df.columns and not y in df.columns else\
                                            (f"{x} not in DataFrame" if not x in df.columns else
                                             f"{y} not in DataFrame")
        print(msg)
        return None
    
    fit = np.linalg.lstsq(np.vstack([df[x], np.ones(len(df[x]))]).T, df[y], rcond=None)
    return fit

def regress_prop(df, x, y, prop, k=0, build=True, prefix=None, show=False, loc=None, save=False):
    fit = regression(df, x, y)

    # prop build is a series of scalars, prop hit is a series of booleans
    pline = prop_build(prop, k, df) if build else prop_hit(prop, k, df)
    if show:
        fig, ax = plt.subplots()
        ax.plot(df[x], fit[0][0]*df[x] + fit[0][1], color='Black', label="Regression Line")
        ax.plot(df[x], [df[y].mean()]*len(df[x]), color='Blue', label=f"{y} Mean")
        ax.plot([df[x].mean()]*len(df[y]), df[y], color='Orange', label=f"{x} Mean")

        # if prop line is a series of booleans, color by True/False
        # if prop line is a series of scalars, adjust size by scalar value and color by sign
        if build:
            psize = 1 + abs(pline)
            ax.scatter(df[x], df[y], s=psize, color=np.where(pline > k, 'Green', 'Red'))
        else:
            ax.scatter(df[x], df[y], color=np.where(pline==True, 'Green', 'Red'))
            
        fname = f'{y}_vs_{x}_with_{prop}_regression.png'
        if prefix != None:
            fname = prefix + "_" + fname
            ax.set_title(f'{prefix} {y} vs {x}; {prop} > {k}')
            ax.legend()
        if loc != None:
            fname = loc + "/" +fname
        if save:
            fig.savefig(fname)
        plt.show()
    return fit

def chiSquare(x, y):
    quads = [ [x.sum(), y.sum()], [np.logical_not(x).sum(), np.logical_not(y).sum()]]
    return stats.chisquare(quads)
    # crosstab = pd.crosstab(df[x], df[y])
    # chi2, p, dof, ex = stats.chi2_contingency(crosstab)
    # return chi2, p, dof, ex

def process_games(daysSlate, outputLocation, spreadDict = None):
    prod = {}
    os.makedirs(outputLocation, exist_ok=True)
    # itereate through each game
    for gm in daysSlate:
        # unpack keys
        t1, t2 = gm.keys()
        # if spreadDict is not none, we're doing regression with prop
        if spreadDict is not None:

            # to get the spread value
            kys = spreadDict.keys()
            # determine spread direction
            tar = None
            sprd = 0
            if t1 in kys:
                d = {t1 : -1*spreadDict[t1], t2 : spreadDict[t1]}
            else:
                d = {t1 : spreadDict[t2], t2 : -1*spreadDict[t2]}

            def innerfunc(t, ax1, ax2):
                fit = regress_prop(gm[t]["team"], "PtsFor", "TotalPts", prop="Spread", k=d[t],
                                build=False, show=False, save=False)
                ax1.scatter(gm[t]["team"]["PtsFor"], gm[t]["team"]["TotalPts"],
                            c = np.where(gm[t]["team"]["Spread"] > d[t], 'Green', 'Red'),
                            s = 1 + abs(gm[t]["team"]["Spread"]))
                ax1.plot(gm[t]['team']['PtsFor'].mean()*np.ones(len(gm[t]['team']['PtsFor'])),
                            gm[t]['team']['TotalPts'], color='Orange', label='PtsFor Mean')
                ax1.plot(gm[t]['team']['PtsFor'], gm[t]['team']['TotalPts'].mean()*np.ones(len(gm[t]['team']['PtsFor'])),
                            color='Blue', label='TotalPts Mean')
                ax1.plot(gm[t]["team"]["PtsFor"], fit[0][0]*gm[t]["team"]["PtsFor"] + fit[0][1], color='Black', label="Regression Line")
                ax1.set_title(t + "\nTotalPts vs PtsFor")
                ax1.legend()
                
                ax2.scatter(gm[t]["team"]["PtsFor"],
                            gm[t]["team"]["TotalPts"] - (fit[0][0]*gm[t]["team"]["PtsFor"] + fit[0][1]),
                            c = np.where(gm[t]["team"]["Spread"] > d[t], 'Green', 'Red'),
                            s = 1 + abs(gm[t]["team"]["Spread"]))
                ax2.plot(gm[t]["team"]["PtsFor"], [0]*len(gm[t]["team"]["PtsFor"]), color='Black')
                ax2.set_title('Residuals')
                ax2.legend()

                return fit, ax1, ax2
            
            fig, axes = plt.subplots(2,2, figsize=(15,7.5))

            for i, tm in enumerate(gm.keys()):
                fit, regax, resax = innerfunc(tm, axes[0][i], axes[1][i])
            
            title = " vs. ".join([t for t in gm.keys()]) + f" with {t1} {d[t1]}"
            fig.suptitle(title)

            fname = "_".join([t for t in gm.keys()]) + f"_with_{tar}_regression.png"
            fig.savefig(f"{outputLocation}/{fname}")

        fig, axes = plt.subplots(1, 2)

        for i, t in enumerate(gm.keys()):
            df = gm[t]["team"]
            smoothFor = df.PtsFor.rolling(5).mean()
            smoothAgainst = df.PtsAgainst.rolling(5).mean()
            idx = pd.to_datetime(df.game_day)
            axes[i].plot(idx, smoothFor, color='Blue', label="PtsFor")
            axes[i].plot(idx, smoothAgainst, color='Red', label="PtsAgainst")
            axes[i].set_title(f'{t}')
            axes[i].legend()
            axes[i].set_xticklabels(labels=[])
        tms = "_".join([t for t in gm.keys()])
        fname = f'{tms}_PtsFor_and_PtsAgainst_TimeSeries.png'
        fig.suptitle("PtsFor and PtsAgainst Time Series")
        
        if outputLocation != None:
            fname = outputLocation + "/" + fname
        
        fig.savefig(fname)
        plt.close(fig)

"""
Build a function to generate a random forest from a function that takes the following parameters:
    - DataFrame
    - List of Features
    - Target Feature
    - Number of Trees
    - Max Depth
    - Min Samples Split
    - Min Samples Leaf
    - Random State
    Return 
    - Random Forest
"""

def build_forest(df, numeric_features, target, categorical_features=None, n_trees=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder


    forest = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf, random_state=random_state)
    
    
    X_num = df[numeric_features]
    y_train = df[target]
    le = LabelEncoder()
    X_cat = df[categorical_features] if categorical_features != None else None

    if categorical_features != None:
        catshp = X_cat.shape
        X_cat_cols = X_cat.columns
        X_cat = X_cat.to_numpy()
        X_cat = X_cat.reshape(-1, 1)
        X_cat = le.fit_transform(X_cat).reshape(*catshp)
        X_cat = pd.DataFrame(X_cat, columns=X_cat_cols)

    X_cat.index = X_num.index
    X_train = pd.concat([X_num, X_cat], axis=1) if categorical_features != None else X_num
    X_train.fillna(0, inplace=True)
    forest.fit(X_train, df[target])

    if categorical_features != None:
        catshp = X_cat.shape
        X_cat_cols = X_cat.columns
        X_cat = X_cat.to_numpy()
        X_cat = X_cat.reshape(-1, 1)
        X_cat = le.inverse_transform(X_cat).reshape(*catshp)
        X_cat = pd.DataFrame(X_cat, columns=X_cat_cols)
        X_cat.index = X_num.index
        X_train = pd.concat([X_num, X_cat], axis=1)

    return forest