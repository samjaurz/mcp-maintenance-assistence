import React, { useState } from 'react';

// El componente recibe una prop, 'onSendMessage', que es una función
// que se llama cuando el usuario envía un mensaje.
const MessageInput = ({ onSendMessage }) => {
  // 1. Usamos el Hook useState para crear una variable de estado
  //    llamada 'message' y una función para actualizarla, 'setMessage'.
  //    El valor inicial es un string vacío.
  const [message, setMessage] = useState('');

  // 2. Esta función maneja el envío del formulario.
  const handleSubmit = (event) => {
    // Evita que la página se recargue, que es el comportamiento
    // por defecto de los formularios HTML.
    event.preventDefault();

    // 3. Verificamos que el mensaje no esté vacío o solo contenga espacios.
    if (message.trim()) {
      // 4. Llamamos a la función `onSendMessage` que se nos pasó como prop,
      //    enviándole el valor del mensaje.
      onSendMessage(message);
      
      // 5. Reiniciamos el estado del input a un string vacío.
      setMessage('');
    }
  };

  // 6. Esta función actualiza el estado 'message' cada vez que el usuario
  //    escribe algo en el input.
  const handleInputChange = (event) => {
    setMessage(event.target.value);
  };

  return (
    <form onSubmit={handleSubmit} className="message-input-form">
      <input
        type="text"
        value={message}
        onChange={handleInputChange}
        placeholder="Escribe un mensaje..."
        className="message-input"
      />
      <button type="submit" className="send-button">
        Enviar
      </button>
    </form>
  );
};

export default MessageInput;