import React from 'react';

const Message = ({ text, sender }) => {
  // Aquí puedes decidir cómo renderizar el mensaje basado en quién lo envió
  const messageClass = sender === 'user' ? 'message-sent' : 'message-received';
  
  return (
    <div className={`message ${messageClass}`}>
      <div className="message-content">
        <p className="message-text">{text}</p>
        <span className="message-sender">{sender}</span>
      </div>
    </div>
  );
};

export default Message;