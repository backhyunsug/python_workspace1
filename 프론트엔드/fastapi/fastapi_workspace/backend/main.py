from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 애플리케이션 초기화
app = FastAPI() #객체를 만들면 웹서버만들어짐

# CORS(Cross-Origin Resource Sharing) 설정
# 풀스택개발 - 프론트앤드(사용자인터페이스, 눈에보이는 부분만 담당, html, css,javascript-react등)
#             백앤드-Restful API 서버 - 데이터를 보통 json형태로 응답해주는 서버이다. 
#                  fastapi등, 장고, 플라스크, 스프링, php, nodejs 등 ..........
#            백앤드와 프론트앤드는 별도의 서버이다. 서로간에 자바스크립트를 통해 데이터를 주고 ]
#            주고 받아야 하는데 서로 다른 사이트 (도메인, 아이피, 포트번호만 달라도 서로 다른사이트이다)
#            CORS(왜 남의 사이트에서 내 사이트를 오는데 라는 오류임)
#            백앤드에서 이문제를 해결해줘야 한다. 특정 아이피나 도메인 그리고 포트번호는 내정보를 가져
#            갈수있어라고 허락을 해줘야 한다.  CORSMiddleware를 통해서 이걸 열어준다 
#            프론트쪽에 프록시 서버를 만들어서 접근 가능하다 => react는 아예 허용 안함, vuejs는 아직
#            허용중이긴 한데 고려하지 말자. 쓰면 안된다. 보안이 취약해짐 
# React 개발 서버(localhost:3000 등)와 통신하기 위해 필요합니다.
origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://www.sessac.com:5173"
]
#미들웨어 - 중간에 거쳐간다.   클라이언트 ========> 미들웨어1 =>미들웨어2=>미들웨어3 ...=> 서버

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

#라우터 연결하기 
from routers import board , score

app.include_router(board.router)  # http://127.0.0.1:8000/board ~~ => board.py가 처리한다  
app.include_router(score.router)

#실행방법
#conda activate backend 
#python -m uvicorn main:app --reload  --port 8000
