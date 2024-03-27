Originally intended to solely be practice for building Dashboards using Python's Dash & Plotly interfaces, this repo also houses 
NCAA basketball analysis scripts that rely on [CBBpy](https://github.com/dcstats/CBBpy) to retrieve data (Player & Team data stored 
on repo, Play-by-play is too large ~550MB)

NCAA data includes function to get data for both teams that are passed to function via a list of tuples `[(tm1, tm2), (tm3, tm4)...]`
Also there is a `process_games` function that will take the result of the above function and a dictionary of spread values `{ (tm1 OR tm2) : (+/- k) }`
to create a series of regression scatter plots that have color variable for covering the spread and size variable for amount spread covered/missed by.

Currently working on implementing appropriate methods from sci-kit learn to forecast betting results for games
Absolutely need to include docstrings for pertinent methods, clean up `NCAA.ipynb`
Planning to split into 2 repos—one for NCAA & another for NBA. If to occur, likely to be after NCAA Elite 8 round 2024
