from fastapi import FastAPI 
from fastapi.responses import JSONResponse 

from fastapi import APIRouter, Depends 

router = APIRouter(
    prefix="/score",  #url요청이 /board/~~~~ 로 오는것은 여기서 다 처리한다는 의미임 
    tags=["score"],     #swager문에 표시될 태그임   
    responses= {404:{'decription':'Not found'}} #예외처리 
)


#더미데이터
scoreList= [
        {"name":"홍길동", "kor":100, "eng":100, "mat":100, "total":300, "avg":100},
        {"name":"임꺽정", "kor":90, "eng":90, "mat":90, "total":270, "avg":90},
        {"name":"장길산", "kor":80, "eng":80, "mat":80, "total":240, "avg":80}
]

@router.get("/scoreList")
def getScoreList():
    #디비가서 데이터 읽어오기 => 
    return {"scoreList":scoreList}

from fastapi import Body 
@router.post("/score/insert" )
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
