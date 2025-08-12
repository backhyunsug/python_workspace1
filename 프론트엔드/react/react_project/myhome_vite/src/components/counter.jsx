import { useState } from 'react'


function Counter() {
  const [count, setCount] = useState(0)

  return (
    <>
      <h1>카운터</h1>
      <h2>{count}</h2>
      <button type="button" onClick={()=>{ setCount(count+1)}}>증가</button>
      <button type="button" onClick={()=>{ setCount(count-1)}}>감소</button>
    </>
  )
}

export default Counter; 
//파일이 다르면, 서로 못주고 받아서 파일내의 함수나 클래스를 외부로 노출시켜야
//다른 파일에서 이걸 사용가능    