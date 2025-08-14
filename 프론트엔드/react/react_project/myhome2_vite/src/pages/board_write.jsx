
import { useState } from "react"
import { useNavigate } from "react-router-dom"
import axios from "axios";

function BoardWrite(){
    const [board, setBoard] = useState({});
    const history = useNavigate();
    const sendServer = async ()=>{
        //버튼으로 받은거라 submit가 아님 preventDefault호출할 필요가 없다. 
        //multipart 데이터 타입을 서버로 보내려면 반드시 FormData 객체로 만들어서 보내야 한다. 
        var frmData = new FormData();
        frmData.append("title", board.title);
        frmData.append("writer", board.writer);
        frmData.append("contents", board.contents);
        //파일은 
        frmData.append("filename", document.forms[0].filename.files[0]);
        let result = await axios.post("http://127.0.0.1:8000/board/insert", frmData)
        alert("등록되었습니다");
        history("/board")
    }

    const {title, writer, contents, filename }= board; //해체-디스터럭션(json)-> 일반변수로 
    const onChange = (e)=>{
        const { value, name } = e.target; // 우선 e.target 에서 name 과 value 를 추출
        setBoard({
        ...board, // 기존의 board 객체를 복사한 뒤
        [name]: value // name 키를 가진 값을 value 로 설정
      });
    }
    
    return(
        <div className="container">
            <h1>게시판</h1>
            <form method="post" encType="multipart/form-data">
                <div className="form-floating mb-3">
                    <input 
                        type="text" 
                        className="form-control" 
                        name="title"
                        value={title}
                        onChange={onChange}
                        placeholder=" "
                        required
                    />
                    <label htmlFor="title">제목</label>
                </div>
                <div className="form-floating mb-3">
                    <input 
                        type="text" 
                        className="form-control"
                        name="writer"
                        value={writer}
                        onChange={onChange}
                        placeholder=" "
                        required
                    />
                    <label htmlFor="writer">작성자</label>
                </div>
                <div className="form-floating mb-3">
                    <textarea
                        className="form-control"
                        name="contents" 
                        value={contents}
                        onChange={onChange}
                        placeholder=" "
                        style={{ height: '150px' }}
                        required
                    ></textarea>
                    
                    <label htmlFor="contents">내용</label>
                </div>
     
              <div className="mb-3">
                    <label htmlFor="filename" className="form-label">파일 첨부</label>
                    <input 
                        type="file"
                        name="filename" 
                        className="form-control"
                        onChange={onChange}
                    />
              </div>

              <div className="d-flex justify-content-end">
                    <button type="button" onClick={()=>{sendServer()}} className="btn btn-primary me-2">등록</button>
                    <button type="button" className="btn btn-secondary" onClick={() => navigate('/board/list')}>취소</button>
              </div>
            </form>
        </div>
    )
}

export default BoardWrite;