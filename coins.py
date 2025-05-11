import requests
import pandas as pd
import numpy as np
import json
def coins_():
    url = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
    response = requests.get(url)
    response = response.json()
    with open ("file.json", "w") as file:
        json.dump(response, file, indent=2, ensure_ascii=False)
    coins = []
    count = 0
    if "data" in response:
        for i in response['data']:
            if count != 10:
                coins.append(f"{i.get('instId')}T")
                count+=1
            else:
                break
    coins.append("SUI-USDT")
    return coins
# resp = coins_()
# print(resp)