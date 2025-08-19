import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 


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
    prefix="/predict",  #url요청이 /board/~~~~ 로 오는것은 여기서 다 처리한다는 의미임 
    tags=["predict"],   #swager문에 표시될 태그임   
    responses= {404:{'decription':'Not found'}} #예외처리 
)


class ImageClasifier (nn.Module):
    def __init__(self, input_size=28*28, hidden_size=500, num_classes=10):
        #그림크기가 28 by 28 
        super(ImageClasifier, self).__init__() #첫번째인자 클래스명, 두번째인자:객체자신 
        self.fc1 =  nn.Linear(input_size, hidden_size) #784 -> 500
        self.relu = nn.ReLU() 
        self.fc2 =  nn.Linear(hidden_size, num_classes) #500 -> 10개로 

    def forward(self, x): 
        #이미지를 1차원 벡터로 만들어서 전달해야 한다 
        x = x.reshape(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x 
    
#이미지 하나를 읽어서 예측하기 
from PIL import Image  #이미지 처리 라이브러리  
def predict_image(image_path):#이미지 경로를 입력받는다 
    #모델 불러오기 
    try:
        # 모델 구조를 다시 정의하고 가중치를 로드하거나, 전체 모델을 로드
        # 여기서는 전체 모델을 로드하는 방식으로 진행
        # 1. 모델 클래스 인스턴스 생성 (필수)
        loaded_model = ImageClasifier() #fastapi 쪽에 클래스가 있어야 한다 
        # 2. state_dict 로드
        loaded_model.load_state_dict(torch.load("mnist.pth"))
    except FileNotFoundError:
        print("Error: The trained model file 'mnist.pth' was not found.")
        exit()

    # 1. 이미지 불러오기
    try:
        image = Image.open(image_path).convert('L')  # 'L' 모드로 변환하여 흑백 이미지로 만듦
    except FileNotFoundError:
        print(f"Error: The file at {image_path} was not found.")
        return None

    # 2. 이미지 전처리
    # 모델 학습 시 사용한 것과 동일한 전처리 과정을 적용
    transform = transforms.Compose([
        transforms.Resize((28, 28)),          # 이미지를 28x28 크기로 조정
        transforms.ToTensor(),                # PIL 이미지를 PyTorch 텐서로 변환 (0~1)
        transforms.Normalize((0.5,), (0.5,))  # 텐서를 정규화
    ])


    
    # 전처리된 이미지를 텐서로 변환
    image_tensor = transform(image)
    
    # 3. 모델 입력 형태로 변경
    # DataLoader는 배치 차원을 자동으로 추가하지만, 여기서는 직접 추가해야 함
    image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 28, 28]

    # 4. 모델로 예측
    # 모델을 추론 모드로 전환
    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model(image_tensor)
        # Softmax를 적용하여 확률로 변환
        probabilities = torch.nn.functional.softmax(output, dim=1)
        # 가장 높은 확률을 가진 클래스(숫자)와 그 확률을 찾음
        _, predicted_class = torch.max(probabilities, 1)
        predicted_prob = probabilities[0][predicted_class].item()
        
    if predicted_prob is not None:
        print(f"이미지 '{image_path}'의 예측 결과: {predicted_class}")
        print(f"예측 확률: {predicted_prob:.4f}")

    return predicted_class, predicted_prob  ##########################


@router.get("/")
def board_index():
    return {"msg":"예측하기"}

@router.post("/insert")
def board_insert( 
    filename:Optional[UploadFile] = File(None),
    title:str = Form(...), #Form 객체를 통해서 정보를 받는다. ...:필수필드
    writer:str = Form(...),
    contents:str = Form(...),
    settings:dict = Depends(get_settings)
):
    temp_filename=""
    temp_image_url=""

    #파일이 업로드 먼저 처리하기 
    predict_class="예측불가"
    predict_prob=0 

    if filename and filename.filename: 
        file_location = os.path.join(settings["UPLOAD_DIRECTORY"], 
                                     filename.filename) 
        #클라이언트로부터 파일을 받아온다. 이때 모든정보는 filename 
        #객체로 받아옴 이 객체는 filename 속성도 있고, file정보속성도 있고 
        #확인해서 파일정보가 맞지 않으면 정지시키거나 지나치게 용량이 커도 
        #안된다. 용량확인도 해줘야 하는데  copyfileobj 를 통해서 서버폴더에 저장함
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(filename.file, buffer)
        temp_filename=filename.filename 
        temp_image_url = "static/" + filename.filename
        print()
        predict_class, predict_prob = predict_image(file_location)
        predict_class = predict_class.item() 
        #tensor([5])
        file_response = f'파일 {filename.filename} 가 업로드되었습니다'
    else:
        file_response = "파일이 첨부되지 않았습니다."
    
    sql = """
      insert into tb_board (title, writer, contents, filename, image_url,
      wdate, hit) values(:title, :writer, :contents, :filename, :image_url,
      now(), 0)
      """
    with Database() as db_mgr:
        data =[{"title":title, "writer":writer, "contents":contents, 
                "filename":temp_filename, "image_url":temp_image_url}]
        db_mgr.execute(sql, data)


    return {"msg":"등록성공", "클래스":predict_class, "확률":predict_prob}



