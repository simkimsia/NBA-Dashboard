import re
import pandas as pd

def map_matchup(x):
    homeMatch = re.compile("(\w+) vs. (\w+)")
    awayMatch = re.compile("(\w+) @ (\w+)")
    if homeMatch.match(x):
        return ("Home", homeMatch.match(x).groups()[-1])
    elif awayMatch.match(x):
        return ("Away", awayMatch.match(x).groups()[-1])
    else:
        return None
    
def map_wl(x):
    if x == "W":
        return 1
    elif x == "L":
        return 0
    else:
        return None
    
def map_gamedate(x):
    return pd.to_datetime(x)

def clean_df(df):
    if "MATCHUP" in df.columns:
        res = df.MATCHUP.map(map_matchup)
        df["HomeAway"] = res.map(lambda x: x[0])
        df["Opponent"] = res.map(lambda x: x[1])
        df.drop(columns=["MATCHUP"], inplace=True, axis=1)
    elif "HomeAway" in df.columns and "Opponent" in df.columns:
        if "MATCHUP" in df.columns:
            df.drop(columns=["MATCHUP"], inplace=True, axis=1)
    else:
        print("Issue with DataFrame:\nMissing MATCHUP column\nNeed to re-query API")
    # df["WL"] = df["WL"].map(map_wl)
    df["GameDate"] = df["GAME_DATE"].map(map_gamedate)
    return df