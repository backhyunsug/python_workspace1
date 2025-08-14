import {Routes, Route, Link, Outlet} from 'react-router-dom'
import 'bootstrap/dist/css/bootstrap.min.css'

function Head(){
    //자바스크립트가 class가 예약어임 jsx문법에서 className -> classNameName으로 바꿔써라
    //a 태그는 페이지 자체를 서버로 부터 다시 가져온다. SPA(Single Page Application)는 페이지 새로 
    //고침 하지 않고 비동기적으로 가져오면 알아서 다시 랜더링 Link 
    return (
<div className='container-fluid'>
    <nav className="navbar navbar-expand-sm bg-dark navbar-dark">
        <div className="container-fluid">
          <Link className="navbar-brand" >Logo</Link>
          <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#collapsibleNavbar">
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="collapsibleNavbar">
            <ul className="navbar-nav">
              <li className="nav-item">
                <Link className="nav-link"  to="/">홈</Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/board">게시판</Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" >Link</Link>
              </li>  
              <li className="nav-item dropdown">
                <Link className="nav-link dropdown-toggle"  role="button" data-bs-toggle="dropdown">Dropdown</Link>
                <ul className="dropdown-menu">
                  <li><Link className="dropdown-item" >Link</Link></li>
                  <li><Link className="dropdown-item" >Another link</Link></li>
                  <li><Link className="dropdown-item" >A third link</Link></li>
                </ul>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    </div>
    )
}

export default Head;