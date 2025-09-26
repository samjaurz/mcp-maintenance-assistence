"use client";
import { useState } from "react";
import axiosInstance from "@/services/axiosInstance";

export default function Home() {
  const [text, setText] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] =useState("");

  const handleSend = async () => {

    setQuestion(text);
    try {
      const response = await axiosInstance.post("/send-text", { text });
      setAnswer(response.data.message);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">

      {question && (
        <div className="bg-blue-200 text-blue-800 p-4 rounded-lg w-full max-w-xl mb-4 h-auto overflow-y-auto whitespace-pre-wrap">
          <p className="font-bold">Pregunta:</p>
          {question}
        </div>
      )}

      {answer && (
        <div className="bg-gray-200 text-gray-800 p-4 rounded-lg w-full max-w-xl mb-4 h-auto overflow-y-auto whitespace-pre-wrap">
          <p className="font-bold">Respuesta:</p>
          {answer}
        </div>
      )}
      <input
        type="text"
        placeholder="Ask anything"
        value={text}
        onChange={(e) => setText(e.target.value)}
        className="border rounded px-3 py-2 mb-4 w-80"
      />
      <button
        onClick={handleSend}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >Enviar
      </button>
    </div>
  );
}
