"use client"
import React, { useState, useEffect, useRef } from "react";
import axiosInstance from "@/services/axiosInstance";
import { Manual } from "@/types/manuals";
import { CloudArrowUpIcon, TrashIcon, XMarkIcon, CheckCircleIcon, XCircleIcon } from "@heroicons/react/24/outline";

type DisplayManual = Manual & {
  status: 'completed' | 'processing'; 
};

const Spinner: React.FC = () => (
  <div className="w-5 h-5 border-4 border-blue-400 border-dashed rounded-full animate-spin"></div>
);

const NotificationToast: React.FC<{
  message: string;
  type: 'success' | 'error';
  onClose: () => void;
}> = ({ message, type, onClose }) => {
  const isSuccess = type === 'success';
  const bgColor = isSuccess ? 'bg-green-100 border-green-400' : 'bg-red-100 border-red-400';
  const textColor = isSuccess ? 'text-green-800' : 'text-red-800';
  
  return (
    <div className={`fixed bottom-5 right-5 z-50 p-4 rounded-lg border shadow-lg flex items-center ${bgColor} ${textColor}`}>
      {isSuccess ? <CheckCircleIcon className="h-6 w-6 mr-3" /> : <XCircleIcon className="h-6 w-6 mr-3" />}
      <span className="flex-grow">{message}</span>
      <button onClick={onClose} className="ml-4 p-1 rounded-full hover:bg-black/10">
        <XMarkIcon className="h-5 w-5" />
      </button>
    </div>
  );
};


// --- Componente Principal ---

const ManualDisplay: React.FC = () => {
  const [displayManuals, setDisplayManuals] = useState<DisplayManual[]>([]);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [notification, setNotification] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const fetchManuals = async () => {
    try {
      const response = await axiosInstance.get<Manual[]>(`/manuals/`);
      const manualsData = Array.isArray(response.data) ? response.data : [];
      const formattedManuals = manualsData.map(manual => ({ ...manual, status: 'completed' as const }));
      setDisplayManuals(formattedManuals);
    } catch (error) {
      console.error("Error al obtener los manuales:", error);
    }
  };

  useEffect(() => {
    fetchManuals();
  }, []);
  
  // Efecto para limpiar la notificación después de un tiempo
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => {
        setNotification(null);
      }, 3000); // La notificación desaparece después de 3 segundos
      return () => clearTimeout(timer);
    }
  }, [notification]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const tempId = crypto.randomUUID();
    const tempManual: DisplayManual = {
      id: tempId,
      name: file.name,
      status: 'processing',
    };

    // Muestra el manual temporal con el spinner inmediatamente
    setDisplayManuals(prevManuals => [tempManual, ...prevManuals]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Espera a que la subida se complete
      await axiosInstance.post("/manuals/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // Si tiene éxito, muestra la notificación y recarga la lista
      setNotification({ message: "Upload exitoso", type: 'success' });
      fetchManuals(); // Recarga la lista para obtener el manual real desde el servidor

    } catch (uploadError) {
      console.error("Error en la subida:", uploadError);
      setNotification({ message: "Fallo el upload", type: 'error' });
    } finally {
      // En cualquier caso (éxito o fallo), elimina el manual temporal
      setDisplayManuals(prev => prev.filter(m => m.id !== tempId));
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDelete = async (manualId: string) => {
    // ... la lógica de eliminación no cambia ...
    if (deletingId) return;
    if (!window.confirm("¿Estás seguro de que quieres eliminar este manual?")) return;

    setDeletingId(manualId);
    try {
      await axiosInstance.delete(`/manuals/${manualId}`);
      setDisplayManuals(prev => prev.filter(m => m.id !== manualId));
    } catch (error) {
      console.error("Error al eliminar el manual:", error);
      alert("No se pudo eliminar el manual.");
    } finally {
      setDeletingId(null);
    }
  };


  return (
    <div className="bg-white shadow-sm border border-gray-200 flex flex-col h-full rounded-lg overflow-hidden">
      <div className="flex justify-between items-center px-6 py-4 border-b border-gray-200">
        <h1 className="text-xl font-semibold text-gray-800">Lista de Manuales</h1>
        <div>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            className="hidden"
            accept=".pdf"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 rounded-lg bg-blue-100 shadow-md text-black hover:bg-blue-200 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-300 flex items-center gap-2"
          >
            <CloudArrowUpIcon className="h-6 w-6 text-blue-600" />
            <span>Subir Manual</span>
          </button>
        </div>
      </div>

      <div className="overflow-x-auto h-full">
        <table className="min-w-full text-sm text-left text-gray-700">
          <thead className="bg-gray-50 text-xs text-gray-700 uppercase">
            <tr>
              <th scope="col" className="px-6 py-3">
                Nombre del archivo
              </th>
              <th scope="col" className="px-6 py-3 w-24 text-center">
                Acciones
              </th>
            </tr>
          </thead>
          <tbody>
            {displayManuals.length === 0 ? (
                <tr>
                    <td colSpan={2} className="text-center py-10 text-gray-500">No hay manuales para mostrar.</td>
                </tr>
            ) : (
                displayManuals.map((manual) => (
                  <tr key={manual.id} className="bg-white border-b hover:bg-gray-50">
                    <td className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap">
                      {manual.name}
                    </td>
                    <td className="px-6 py-4 text-center">
                      {manual.status === 'processing' ? (
                        <div className="flex justify-center">
                          <Spinner />
                        </div>
                      ) : (
                        <button
                          onClick={() => handleDelete(manual.id)}
                          disabled={!!deletingId}
                          className="p-1 text-gray-500 hover:text-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
                          aria-label={`Eliminar manual ${manual.name}`}
                        >
                          {deletingId === manual.id ? <Spinner /> : <TrashIcon className="h-5 w-5" />}
                        </button>
                      )}
                    </td>
                  </tr>
                ))
            )}
          </tbody>
        </table>
      </div>

      {/* Renderizar la notificación si existe */}
      {notification && (
        <NotificationToast
          message={notification.message}
          type={notification.type}
          onClose={() => setNotification(null)}
        />
      )}
    </div>
  );
};

export default ManualDisplay;