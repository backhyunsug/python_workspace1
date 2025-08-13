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

from fastapi import Body
import sqlalchemy 
@router.post("/insert")
def board_insert(title:str=Body(...), writer:str=Body(...), contents:str=Body(...)):
    sql = """
        insert into tb_board (title, writer,contents, wdate, hit)
        value(:title,  :writer, :contents, now(), 0)
    """
    params = [{"title":title, "writer":writer, "contents":contents}]
    try:
        with Database() as db_mgr:
            db_mgr.execute(sql, params)
        return {"msg":"등록성공"}
    except sqlalchemy.exc.SQLAlchemy as e:
        return {"msg":"데이터 등록실패"}

