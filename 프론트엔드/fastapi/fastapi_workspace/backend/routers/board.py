from fastapi import FastAPI 
from fastapi.responses import JSONResponse 

from fastapi import APIRouter, Depends 
from database import Database

router = APIRouter(
    prefix="/board",  #url요청이 /board/~~~~ 로 오는것은 여기서 다 처리한다는 의미임 
    tags=["board"],     #swager문에 표시될 태그임   
    responses= {404:{'decription':'Not found'}} #예외처리 
)

@router.get("/")
def board_index():
    return {"msg":"게시판입니다"}

@router.get("/list")
def board_list():
    with Database() as db_mgr:
        sql = "select * from tb_board" 
        results = db_mgr.executeAll(sql)
    return {"list":results}


