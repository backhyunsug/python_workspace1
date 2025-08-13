from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 애플리케이션 초기화
app = FastAPI()

# CORS(Cross-Origin Resource Sharing) 설정
# React 개발 서버(localhost:3000 등)와 통신하기 위해 필요합니다.
origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)


#@-데코레이터   app
@app.get("/")
def index():
    return {"message":"Hello FastAPI"}

#값받기 
#http://127.0.0.1:8000/add?x=4&y=8
@app.get("/add")
def add(x:int, y:int):
    return {"x":x, "y":y, "result":x+y}

#http://127.0.0.1:8000/add2/x/y
@app.get("/add2/{x}/{y}")
def add(x:int, y:int):
    return {"x":x, "y":y, "result":x+y}


#더미데이터
scoreList= [
        {"name":"홍길동", "kor":100, "eng":100, "mat":100, "total":300, "avg":100},
        {"name":"임꺽정", "kor":90, "eng":90, "mat":90, "total":270, "avg":90},
        {"name":"장길산", "kor":80, "eng":80, "mat":80, "total":240, "avg":80}
]

@app.get("/scoreList")
def getScoreList():
    #디비가서 데이터 읽어오기 => 
    return {"scoreList":scoreList}

from fastapi import Body 
@app.post("/score/insert" )
def score_insert(name:str = Body(..., embed=True),
                 kor:int = Body(..., embed=True),
                 eng:int = Body(..., embed=True),
                 mat:int = Body(..., embed=True) 
                 ):
    score = {"name":name, 
             "kor":kor, 
             "eng":eng, 
             "mat":mat,
             "total":kor+eng+mat,
             "avg":(kor+eng+mat)/3}
    scoreList.append(score)
    return score 




#실행방법 
#conda activate backend
#python -m uvicorn main:app  --reload 
#http://127.0.0.1:8000/docs

