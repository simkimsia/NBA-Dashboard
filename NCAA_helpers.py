from scipy import stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
pd.options.mode.copy_on_write = True

team_df = pd.read_csv('NCAAteams.csv')
plyr_df = pd.read_csv('NCAAplyrs.csv')

def update_data():
    team_df = pd.read_csv('NCAAteams.csv')
    plyr_df = pd.read_csv('NCAAplyrs.csv')

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
            return None
            
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
                d = {t1 : spreadDict[t1], t2 : -1*spreadDict[t1]}
            else:
                d = {t1 : -1*spreadDict[t2], t2 : spreadDict[t2]}

            

            # if t1 is in spreadDict, t1 is the target
            if t1 in kys:
            
                tar = t1
                sprd = spreadDict[t1]
            
                # begin dependence on spread direction
                t1_ForVsTotal = regress_prop(gm[t1]["team"], "PtsFor", "TotalPts",
                                                prop="Spread", k=sprd,
                                                build=False, show=False,
                                                save=False)
            
                t2_ForVsTotal = regress_prop(gm[t2]["team"], "PtsFor", "TotalPts", 
                                                prop="Spread", k=-1*sprd,
                                                build=False, show=False,
                                                save=False)
                # end dependence on spread direction

                fig, axes = plt.subplots(1,2, figsize=(15,7.5))
                axes[0].plot(gm[t1]["team"]["PtsFor"],
                             t1_ForVsTotal[0][0]*gm[t1]["team"]["PtsFor"] + t1_ForVsTotal[0][1], 
                             color='Black', label="Regression Line")
                
                # begin dependence on spread direction
                axes[0].scatter(gm[t1]["team"]["PtsFor"], 
                                gm[t1]["team"]["TotalPts"],
                                c = np.where(gm[t1]["team"]["Spread"] > sprd, 'Green', 'Red'),
                                s = 1 + abs(gm[t1]["team"]["Spread"]))
                axes[0].plot(gm[t1]['team']['PtsFor'].mean()*np.ones(len(gm[t1]['team']['PtsFor'])),
                                gm[t1]['team']['TotalPts'], color='Orange', label='PtsFor Mean')
                axes[0].plot(gm[t1]['team']['PtsFor'], gm[t1]['team']['TotalPts'].mean()*np.ones(len(gm[t1]['team']['PtsFor'])),
                                color='Blue', label='TotalPts Mean')
                axes[0].set_title(f'{t1} PtsFor vs TotalPts; Spread > {sprd}')
                axes[0].legend()
                axes[1].plot(gm[t2]["team"]["PtsFor"],
                             t2_ForVsTotal[0][0]*gm[t2]["team"]["PtsFor"] + t2_ForVsTotal[0][1], 
                             color='Black', label="Regression Line")
                axes[1].scatter(gm[t2]["team"]["PtsFor"], 
                                gm[t2]["team"]["TotalPts"],
                                c = np.where(gm[t2]["team"]["Spread"] > -1*sprd, 'Green', 'Red'),
                                s = 1 + abs(gm[t2]["team"]["Spread"]))
                axes[1].set_title(f'{t2} PtsFor vs TotalPts; Spread > {-1*sprd}')
                axes[1].plot(gm[t2]['team']['PtsFor'].mean()*np.ones(len(gm[t2]['team']['PtsFor'])),
                                gm[t2]['team']['TotalPts'], color='Orange', label='PtsFor Mean')
                axes[1].plot(gm[t2]['team']['PtsFor'], gm[t2]['team']['TotalPts'].mean()*np.ones(len(gm[t2]['team']['PtsFor'])),
                                color='Blue', label='TotalPts Mean')
                # end dependence on spread direction

                fig.suptitle(f'{t1} vs {t2} with {tar} > {sprd}')

                fname = f'{t1}_vs_{t2}_with_{tar}_regression.png'
                
                if outputLocation != None:
                    fname = outputLocation + "/" +fname
                
                fig.savefig(fname)
                plt.close(fig)
                # plot residuals
                fig, axes = plt.subplots(1,2, figsize=(15,7.5))
                axes[0].scatter(gm[t1]["team"]["PtsFor"],
                                gm[t1]["team"]["TotalPts"] - (t1_ForVsTotal[0][0]*gm[t1]["team"]["PtsFor"] + t1_ForVsTotal[0][1]),
                                c = np.where(gm[t1]["team"]["Spread"] > sprd, 'Green', 'Red'),
                                s = 1 + abs(gm[t1]["team"]["Spread"]))
                axes[0].set_title(f'{t1} TotalPts vs PtsFor Residuals')
                axes[0].plot(gm[t1]["team"]["PtsFor"], [0]*len(gm[t1]["team"]["PtsFor"]), color='Black')
                axes[1].scatter(gm[t2]["team"]["PtsFor"],
                                gm[t2]["team"]["TotalPts"] - (t2_ForVsTotal[0][0]*gm[t2]["team"]["PtsFor"] + t2_ForVsTotal[0][1]),
                                c = np.where(gm[t2]["team"]["Spread"] > -1*sprd, 'Green', 'Red'),
                                s = 1 + abs(gm[t2]["team"]["Spread"]))
                axes[1].set_title(f'{t2} TotalPts vs PtsFor Residuals')
                axes[1].plot(gm[t2]["team"]["PtsFor"], [0]*len(gm[t2]["team"]["PtsFor"]), color='Black')

                fig.suptitle = f'{t1} vs {t2} with {tar} > {sprd} Residuals'

                fname = f'{t1}_vs_{t2}_with_{tar}_residuals.png'
                
                if outputLocation != None:
                    fname = outputLocation + "/" +fname
                
                fig.savefig(fname)
                plt.close(fig)
            elif t2 in kys:
                tar = t2
                sprd = spreadDict[t2]
                t1_ForVsTotal = regress_prop(gm[t1]["team"], "PtsFor", "TotalPts",
                                                prop="Spread", k=-1*sprd,
                                                build=False, show=False,
                                                save=False, prefix=t1)
                t2_ForVsTotal = regress_prop(gm[t2]["team"], "PtsFor", "TotalPts", 
                                                prop="Spread", k=sprd,
                                                build =False, prefix=t2)
                fig, axes = plt.subplots(1,2, figsize=(15,7.5))
                axes[0].plot(gm[t1]["team"]["PtsFor"],
                             t1_ForVsTotal[0][0]*gm[t1]["team"]["PtsFor"] + t1_ForVsTotal[0][1], 
                             color='Black', label="Regression Line")
                axes[0].scatter(gm[t1]["team"]["PtsFor"], 
                                gm[t1]["team"]["TotalPts"],
                                c = np.where(gm[t1]["team"]["Spread"] > -1*sprd, 'Green', 'Red'),
                                s = 1 + abs(gm[t1]["team"]["Spread"]))
                axes[0].plot(gm[t1]['team']['PtsFor'].mean()*np.ones(len(gm[t1]['team']['PtsFor'])),
                                gm[t1]['team']['TotalPts'], color='Orange', label='PtsFor Mean')
                axes[0].plot(gm[t1]['team']['PtsFor'], gm[t1]['team']['TotalPts'].mean()*np.ones(len(gm[t1]['team']['PtsFor'])),
                                color='Blue', label='TotalPts Mean')
                axes[0].set_title(f'{t1} PtsFor vs TotalPts; Spread > {-1*sprd}')
                axes[0].legend()
                axes[1].plot(gm[t2]["team"]["PtsFor"],
                             t2_ForVsTotal[0][0]*gm[t2]["team"]["PtsFor"] + t2_ForVsTotal[0][1], 
                             color='Black', label="Regression Line")
                axes[1].scatter(gm[t2]["team"]["PtsFor"], 
                                gm[t2]["team"]["TotalPts"],
                                c = np.where(gm[t2]["team"]["Spread"] > sprd, 'Green', 'Red'),
                                s = 1 + abs(gm[t2]["team"]["Spread"]))
                axes[1].set_title(f'{t2} PtsFor vs TotalPts; Spread > {sprd}')
                axes[1].plot(gm[t2]['team']['PtsFor'].mean()*np.ones(len(gm[t2]['team']['PtsFor'])),
                                gm[t2]['team']['TotalPts'], color='Orange', label='PtsFor Mean')
                axes[1].plot(gm[t2]['team']['PtsFor'], gm[t2]['team']['TotalPts'].mean()*np.ones(len(gm[t2]['team']['PtsFor'])),
                                color='Blue', label='TotalPts Mean')
                
                fig.suptitle(f'{t1} vs {t2} with {tar} > {sprd}')

                fname = f'{t1}_vs_{t2}_with_{tar}_regression.png'

                if outputLocation != None:
                    fname = outputLocation + "/" +fname

                fig.savefig(fname)
                plt.close(fig)
                # plot residuals
                fig, axes = plt.subplots(1,2, figsize=(15,7.5))
                axes[0].scatter(gm[t1]["team"]["PtsFor"],
                                gm[t1]["team"]["TotalPts"] - (t1_ForVsTotal[0][0]*gm[t1]["team"]["PtsFor"] + t1_ForVsTotal[0][1]),
                                c = np.where(gm[t1]["team"]["Spread"] > -1*sprd, 'Green', 'Red'),
                                s = 1 + abs(gm[t1]["team"]["Spread"]))
                axes[0].set_title(f'{t1} TotalPts vs PtsFor Residuals')
                axes[0].plot(gm[t1]["team"]["PtsFor"], [0]*len(gm[t1]["team"]["PtsFor"]), color='Black')
                axes[1].scatter(gm[t2]["team"]["PtsFor"],
                                gm[t2]["team"]["TotalPts"] - (t2_ForVsTotal[0][0]*gm[t2]["team"]["PtsFor"] + t2_ForVsTotal[0][1]),
                                c = np.where(gm[t2]["team"]["Spread"] > sprd, 'Green', 'Red'),
                                s = 1 + abs(gm[t2]["team"]["Spread"]))
                axes[1].set_title(f'{t2} TotalPts vs PtsFor Residuals')
                axes[1].plot(gm[t2]["team"]["PtsFor"], [0]*len(gm[t2]["team"]["PtsFor"]), color='Black')

                fig.suptitle = f'{t1} vs {t2} with {tar} > {sprd} Residuals'

                fname = f'{t1}_vs_{t2}_with_{tar}_residuals.png'

                if outputLocation != None:
                    fname = outputLocation + "/" +fname

                fig.savefig(fname)
                plt.close(fig)
        else:
            t1_ForVsTotal = regression(t1, "PtsFor", "TotalPts")
            t2_ForVsTotal = regression(t2, "PtsFor", "TotalPts")
        
        prod[(t1,t2)] = (t1_ForVsTotal, t2_ForVsTotal)

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

    return prod