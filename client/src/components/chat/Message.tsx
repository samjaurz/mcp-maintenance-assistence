type MessageProps = {
  role: "user" | "bot"; 
  text: string;
};

function Message({ role , text }: MessageProps) {
  const isUser = role === "user";
  return (
    <div
      className={`mb-2 p-2 rounded max-w-xs ${
        isUser
          ? "bg-blue-500 text-white self-end ml-auto"
          : "bg-gray-200 text-black"
      }`}
    >
      {text}
    </div>
  );
}

export default Message;
