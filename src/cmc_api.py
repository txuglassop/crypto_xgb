# credit to Chad Thackery, https://www.youtube.com/watch?v=opFegHZ7pUU for tutorial

from requests import Request, Session
import json
from pprint import pprint


url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest'

parameters = {
    'slug':'bitcoin',
    'convert':'USD'
}

headers = {
    'Accepts':'application/json',
    'X-CMC_PRO_API_KEY':'cbc19191-4e16-4e71-a850-b8b8ed8ceef0'
}

session = Session()
session.headers.update(headers)

response = session.get(url, params=parameters)
pprint(json.loads(response.text)['data']['1']['quote']['USD']['price'])