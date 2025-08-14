import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'

import './App.css'

import Home from './pages/home'
import Board from './pages/board'
import BoardWrite from './pages/board_write'
import BoardView from './pages/board_view'

import Head from'./components/head'
import {Routes, Route, Link, Outlet} from 'react-router-dom'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <Head/>

      <Routes>
        <Route path="/" element={<Home/>} />
        <Route path="/board" element={<Board/>} />
        <Route path="/board/write" element={<BoardWrite/>} />
        <Route path="/board/view/:id" element={<BoardView />} />
        { /* /board/view/:id -> useParams라는 훅으로 받아야 한다.*/}
      </Routes>

       
    </>
  )
}

export default App
