import plotly.graph_objects as go
import pandas as pd 
import numpy as np 

#차트그리는 함수 하나씩 만들어서 호출하기 
def create_scatter(): #산포도 
    print("1.산포도") 
    df = pd.DataFrame( {
        'x_data':[1,2,3,4,5,6,7,8,9,10],
        'y_data':[1,2,3,4,5,6,7,8,9,10],
        'size_data':[10,10,40, 20, 50, 60, 10, 10,10,10]
    })

    #차트에 대한 기본정보 , 데이터와 데이터를 출력할때 marker를 사용한다 
    chart1 = go.Scatter( 
        x = df["x_data"],
        y = df["y_data"], 
        mode='markers', 
        marker=dict(size=df["size_data"], color="red", opacity=0.7,  
                    line = {"width":1, "color":'DarkSlateGrey'} ),
        name="산포도"
    )

    #축에 대한 정보를 따로 만든다 
    layout = go.Layout( title='산점도', 
                       xaxis={"title":"X축"} , 
                       yaxis=dict(title="Y축"),
                       hovermode='closest')
    fig = go.Figure(data = [chart1], layout=layout)
    #fig.show() #브라우저가 작동한다 
    fig.write_html("산포도.html")

if __name__ =="__main__":
    create_scatter()

