#%%
"""
API playground — load API keys from a local `.env` file (if present)
to avoid hard-coding secrets in source.

Create a `.env` file at the project root (copy from `.env.example`).
"""

import os
from pathlib import Path
import requests
from dotenv import load_dotenv


# Attempt to load `.env` from repository root (does not overwrite existing env vars)
load_dotenv(Path(__file__).parent / ".env")
#%%
SPORTMONKS_TOKEN = os.getenv("SPORTMONKS_TOKEN")

if not SPORTMONKS_TOKEN:
	raise RuntimeError("SPORTMONKS_TOKEN not set in .env")

headers = {"Authorization": SPORTMONKS_TOKEN}
base_url = "https://api.sportmonks.com/v3/football"


#%%
response = requests.get(
	"https://api.sportmonks.com/v3/football/odds/pre-match",
	headers=headers,
)

data = response.json()
print(list(data.keys()))

#%%
url = "https://football-xg-statistics.p.rapidapi.com/countries/"

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
if not RAPIDAPI_KEY:
	raise RuntimeError("RAPIDAPI_KEY not set in .env")

rapid_headers = {
	"X-RapidAPI-Key": RAPIDAPI_KEY,
	"X-RapidAPI-Host": "football-xg-statistics.p.rapidapi.com",
}

#%%
response = requests.get(url, headers=rapid_headers)
print(response.json())
#%%


