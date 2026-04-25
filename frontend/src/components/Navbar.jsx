import React, { useState, useEffect } from "react";
import logo from "../assets/logo.png";
import { IoMoon } from "react-icons/io5";
import { FiSun } from "react-icons/fi";

const Navbar = () => {
  const [lightMode, setLightMode] = useState(false);
  useEffect(() => {
    if (lightMode) {
      document.documentElement.classList.add("light");
    } else {
      document.documentElement.classList.remove("light");
    }
  }, [lightMode]);
  return (
    <>
      <div className="nav flex items-center justify-between px-[120px] h-[70px] ">
        <div className="logo flex items-center gap-[1px]">
          <img src={logo} alt="Logo" className="h-[75px]" />
          <h3 className="text-[30px] font-bold tracking-tight ">
            ExplainThis
            <span className="bg-gradient-to-bl from-transparent via-cyan-500 to-blue-800 bg-clip-text text-transparent">
              {" "}
              AI
            </span>
          </h3>
        </div>
        <div className="icons flex items-center mr-[80px] ">
          <i onClick={() => setLightMode(!lightMode)} className="icon mood">
            {" "}
            {lightMode ? <FiSun /> : <IoMoon />}
          </i>
        </div>
      </div>
    </>
  );
};

export default Navbar;
