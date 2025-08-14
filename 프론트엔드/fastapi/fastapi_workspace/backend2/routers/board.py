from fastapi import FastAPI 
from fastapi.responses import JSONResponse 
from fastapi import APIRouter, Depends 
from database import Database
import os, shutil 
from typing import Optional #최근에 언어들 경향이 null이나 None값에 대한 처리를 
#절저하게 하기위해 만든 라이브러리 Optional(객체) 이 객체는 할당이 안되었을수도 있다라는의미임
from fastapi import UploadFile, File, Form, HTTPException

#파일업로드 Depends main.py파일과 board.py간에 변수를 주고 받아야 할때 사용할 라이브러리 

settings_container = {}
def get_settings():
    return  settings_container.get("settings")


router = APIRouter(
    prefix="/board",  #url요청이 /board/~~~~ 로 오는것은 여기서 다 처리한다는 의미임 
    tags=["board"],   #swager문에 표시될 태그임   
    responses= {404:{'decription':'Not found'}} #예외처리 
)

@router.get("/")
def board_index():
    return {"msg":"접속성공"}

#디비데이터 들고오기 
@router.get("/list")
def board_list():
    sql = """
    select id, title, writer, date_format(wdate, '%Y-%m-%d') wdate
    , hit, filename, image_url from tb_board
    """
    print(sql)
    with Database() as db_mgr:
        results = db_mgr.executeAll(sql)

    return {"data": results}

#데이터 추가  - 파일업로드받을때는 multipart라는 방식으로 온다. 
#Body X, Pydemic도 안된다.  Form으로 만 받는다
@router.post("/insert")
def board_insert( 
    filename:Optional[UploadFile] = File(None),
    title:str = Form(...), #Form 객체를 통해서 정보를 받는다. ...:필수필드
    writer:str = Form(...),
    contents:str = Form(...),
    settings:dict = Depends(get_settings)
):
    #파일이 업로드 먼저 처리하기 
    if filename and filename.filename: 
        file_location = os.path.join(settings["UPLOAD_DIRECTORY"], 
                                     filename.filename) 
        #클라이언트로부터 파일을 받아온다. 이때 모든정보는 filename 
        #객체로 받아옴 이 객체는 filename 속성도 있고, file정보속성도 있고 
        #확인해서 파일정보가 맞지 않으면 정지시키거나 지나치게 용량이 커도 
        #안된다. 용량확인도 해줘야 하는데  copyfileobj 를 통해서 서버폴더에 저장함
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(filename.file, buffer)
        
        file_response = f'파일 {filename.filename} 가 업로드되었습니다'
    else:
        file_response = "파일이 첨부되지 않았습니다."
    return {"msg":file_response}



#http://127.0.0.1:8000/static/ACD212F9-B13B-4D8A-BA04-6BC6458F650E.jpeg