import React, { useState } from "react";
import "./App.css";
import Navbar from "./components/Navbar";
import { MdOutlineArrowUpward, MdErrorOutline } from "react-icons/md";
import { FiCopy, FiTrash2, FiLoader } from "react-icons/fi";
import { FaCheck } from "react-icons/fa6";
import api from "./api";

const MAX_CHARS = 5000; // maximum limit

const App = () => {
  const [prompt, setPrompt] = useState("");
  const [copied, setCopied] = useState(false);
  const [simplifiedText, setSimplifiedText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [stats, setStats] = useState(null); // to store original and simplified word counts

  const charsLeft = MAX_CHARS - prompt.length; //how many characters are left before hitting MAX_CHARS
  const isOverLimit = charsLeft < 0; // if the number is zero or negative, it means the user has exceeded the limit
  const hasOutput = simplifiedText || loading || error; // if any of these have a h=value then we have sth to show

  const handleSubmit = async () => {
    if (loading || prompt.trim().length < 10 || isOverLimit) return; // prevent submisstion if one of these is true
    setError("");
    setSimplifiedText("");
    setStats(null);

    try {
      setLoading(true); // start loading state
      const response = await api.post("/explain", { text: prompt }); // send the text to the backend
      const { simplified_text, original_word_count, simplified_word_count } =
        response.data; // response
      setSimplifiedText(simplified_text); // store the simplified text in state to display it
      setStats({
        original: original_word_count,
        simplified: simplified_word_count,
      });
    } catch (err) {
      const msg = err?.response?.data?.detail || "Something went wrong."; //get an error message from backend or use the defult one
      setError(msg);
    } finally {
      setLoading(false); // stop loading
    }
  };

  const handleClear = () => {
    setPrompt("");
    setSimplifiedText("");
    setError("");
    setStats(null);
  }; // reset evrything

  const handleCopy = () => {
    if (!simplifiedText) return; // if there is no text to copy, dont do anything
    navigator.clipboard.writeText(simplifiedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000); // reset the copied state after 2 sec
  };

  return (
    <>
      <Navbar />
      <div className="container ">
        <h3 className="text-[40px] md:text-6xl font-bold tracking-tight text-center">
          Complexity,{" "}
          <span
            className="bg-gradient-to-bl from-transparent via-cyan-500 
    to-blue-800 bg-clip-text text-transparent"
          >
            Simplified.
          </span>
        </h3>
        <p className="mt-12 text-[18px] text-[hsla(0, 2%, 66%, 1.00)]  text-center">
          Got a block of text that makes no sense?
          <br /> Drop it here for a clear, human-friendly explanation.
        </p>
        <div className="inputBox">
          <textarea
            onChange={(e) => setPrompt(e.target.value)} // update the prompt state whenever the user types in the textarea
            value={prompt}
            placeholder="Paste your text here..."
            maxLength={MAX_CHARS}
          ></textarea>
          <div className="toolbar">
            <span
              className={`charCounter transition-colors duration-200 ${
                charsLeft < 50
                  ? "text-red-400"
                  : charsLeft < 200
                    ? "text-yellow-400"
                    : "text-[hsla(0,2%,45%,1)]"
              }`}
            >
              {prompt.length} / {MAX_CHARS}
            </span>

            {prompt !== "" && (
              <div className="iconGroup">
                <i
                  onClick={handleClear}
                  title="Clear"
                  className="sendIcon text-[20px] w-[30px] h-[30px] flex items-center justify-center rounded-[50%] transition-all duration-300 hover:opacity-[.8] opacity-50"
                >
                  <FiTrash2 />
                </i>

                <i
                  onClick={!loading && !isOverLimit ? handleSubmit : undefined}
                  className="sendIcon text-[20px] w-[30px] h-[30px] flex items-center justify-center  rounded-[50%]  transition-all duration-300 hover:opacity-[.8] opacity-50"
                >
                  {loading ? (
                    <>
                      <span className="loader" />
                      <FiLoader />
                    </>
                  ) : (
                    <MdOutlineArrowUpward />
                  )}
                </i>
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="flex items-start gap-3 mt-6 p-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 text-[14px] max-w-[700px] w-full">
            <MdErrorOutline className="text-[20px] shrink-0 mt-[2px]" />
            <span>{error}</span>
          </div>
        )}
        {hasOutput && (
          <>
            {stats && (
              <div className="flex gap-6 mt-8 text-[13px] text-[hsla(0,2%,66%,1)]">
                <span>
                  Original:{" "}
                  <strong className="text-gray-400">
                    {stats.original} words
                  </strong>
                </span>
                <span>→</span>
                <span>
                  Simplified:{" "}
                  <strong className="text-cyan-400">
                    {stats.simplified} words
                  </strong>
                </span>
                <span className="text-green-400 font-semibold">
                  {Math.round(
                    ((stats.original - stats.simplified) / stats.original) *
                      100,
                  )}
                  % shorter
                </span>
              </div>
            )}

            <p className="text-[20px] text-[hsla(0, 2%, 66%, 1.00)]  font-[500] mt-[9vh]">
              Here is the simplified version:
            </p>

            <div className="preview">
              <div className="header w-full h-[70px]">
                <h3 className="text-[18px] text-[hsla(0, 2%, 66%, 1.00)] ">
                  Simplified Output
                </h3>
                <div className="icons flex items-center gap-[15px]">
                  <div
                    onClick={handleCopy}
                    className="icon !w-[auto] !p-[12px] flex items-center gap-[8px] "
                  >
                    {copied ? "Copied!" : "Copy"}{" "}
                    {copied ? <FaCheck /> : <FiCopy />}
                  </div>
                </div>
              </div>
              <div className="content">
                {loading ? (
                  <span className="text-[15px] text-[hsla(0,2%,50%,1)] ">
                    Simplifying your text...{" "}
                  </span>
                ) : (
                  simplifiedText || (
                    <span className="text-[hsla(0,2%,40%,1)] italic"></span>
                  )
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
};

export default App;
