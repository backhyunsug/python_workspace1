#뷰티플스프를 이용해서 html파싱 
#로컬에 있는 문서를 읽어서 파싱하는것만 진행한다. 
#weather_table 
import requests 
import json     #지금 오는 데이터는 json 형태의 문자열 이다. 
#서버한테 정보를 보낼때 get, post 방식이 있다 
#누가한테나 허용됨 
url = "https://sports.daum.net/prx/hermes/api/team/rank.json?leagueCode=epl&seasonKey=20252026&page=1&pageSize=100"
#권한없이 남의 사이트 가서 데이터 퍼오면 안됨, 살짝 속이기 가능 

#마치 내가 브라우저를 통해 접근한것처럼 속이기 - 다 먹히는거 아님 
customer_header = {
    "referer":"https://sports.daum.net/",
    "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
}

text=""
response = requests.get(url, headers=customer_header) #그래도 못가져오면 api 찾아봐야 한다 

import pandas as pd 
df = pd.DataFrame(columns=["name", "gp", "gf"])

if response.status_code ==200:
    text = json.loads(response.text) #json.loads 함수를 써서 dict타입으로 바꿔야 한다 
    #print(type(text))    

    dataList = text["list"]
    for item in dataList:
        data = dict() 
        data["name"] = item["name"]
        data["gp"] = item["nameMain"]
        data["gf"] = item["rank"]["gf"]
        df.loc[len(df)] = data 
        print(data) #=>판다스로 바꾸어서 저장을 해보자 

df.to_csv("해외축구.csv", encoding="utf-8-sig", index=False)


#010 9083 7981 
