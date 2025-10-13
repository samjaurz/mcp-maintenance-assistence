"use client";
import { useState } from "react";
import MessageList from "@/components/chat/MessageList";
import ChatInput from "@/components/chat/MessageInput";
import axiosInstance from "@/services/axiosInstance";
import { MessageType } from "@/types/chat";


function ChatContainer() {
  const [messages, setMessages] = useState<MessageType[]>([
    { role: "bot", text: "Ask me something" }
  ]);

  const handleSend = async (text: string) => {
    if (!text.trim()) return;

    const userMsg: MessageType = { role: "user", text };

    setMessages((prev) => [...prev, userMsg]);

    try {
      const response = await axiosInstance.post("/send-text", { text });
      const botMsg: MessageType = { role: "bot", text:  response.data.message };

  
      setTimeout(() => {
        setMessages((prev) => [...prev, botMsg]);
      }, 500);
    } catch (error) {
      console.error("Error al enviar el mensaje:", error);
      const errorMsg: MessageType = { role: "bot", text: "❌ Error al procesar tu mensaje" };
      setMessages((prev) => [...prev, errorMsg]);
    }
  };

  return (
    <div className="flex flex-col w-full "  style={{ height: "calc(100vh - 75px)" }}>
      <MessageList messages={messages} />
      <ChatInput onSend={handleSend} />
    </div>
  );
}

export default ChatContainer;
