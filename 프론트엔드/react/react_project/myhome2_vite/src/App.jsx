import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'

import './App.css'

import Home from './pages/home'
import Board from './pages/board'
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
      </Routes>

       
    </>
  )
}

export default App
