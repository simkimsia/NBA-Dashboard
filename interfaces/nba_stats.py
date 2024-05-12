import time

import pandas as pd
from nba_api.stats.endpoints import playergamelog
from requests.exceptions import ReadTimeout

from NBA_helpers import clean_df


def get_player_game_log(player_id, season="2023"):
    """Fetches player game log data for a specified season."""
    try:
        # Attempt to fetch data
        game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        return game_log.get_data_frames()[0]
    except ReadTimeout:
        # Handle timeouts by logging and returning an empty DataFrame
        print(f"Timeout occurred for player_id: {player_id}")
        return pd.DataFrame()


def fetch_data_with_delays(player_ids):
    """Fetches game logs for multiple players with a delay between each request to prevent rate limits."""
    data = {}
    for player_name, player_id in player_ids.items():
        data[player_name] = get_player_game_log(player_id)
        time.sleep(1)  # Sleep for 1 second between requests to comply with rate limits
    return data


from nba_api.stats.endpoints import teamgamelog  # Example endpoint


def fetch_team_data_with_delays(team_ids):
    # Similar logic to fetch_data_with_delays, but for teams
    data = {}
    for team_name, team_id in team_ids.items():
        try:
            game_log = teamgamelog.TeamGameLog(team_id=team_id, season="2023")
            data[team_name] = clean_df(game_log.get_data_frames()[0])
        except Exception as e:
            print(f"Error fetching data for team {team_name}: {e}")
            data[team_name] = pd.DataFrame()
        time.sleep(1)  # Delay to prevent rate limiting
    return data
