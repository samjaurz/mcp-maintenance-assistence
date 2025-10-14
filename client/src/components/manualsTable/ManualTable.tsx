
import React from 'react';
import { Manual } from '@/types/manuals';
import axiosInstance from "@/services/axiosInstance";

interface ManualsTableProps {
  manuals: Manual[];
}

const ManualTable: React.FC<ManualsTableProps> = ({ manuals }) => {

const handleDeleteManual = async (manualId: number) => {
    const isConfirmed = window.confirm("¿Estás seguro de que quieres eliminar este manual?");
    if (!isConfirmed) return;

    await axiosInstance.delete(`/manuals/${manualId}`);
    alert("Manual eliminado correctamente.");
    
  };


  return (<div className="py-3 px-2 h-full">
    <div className="h-full flex flex-col">
      <table className="border-collapse border border-gray-400 w-full ">
        <thead className="bg-gray-100 sticky top-0 z-10">
          <tr>
            <th className="px-3 py-2 text-center font-medium text-gray-700">Name ↑↓</th>
            <th className="px-4 py-2 text-center font-medium text-gray-700">Date ↑↓</th>
            <th className="px-4 py-2 text-center font-medium text-gray-700">Actions</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {manuals && manuals.length > 0 ? (
            manuals.map((manual) => (
              <tr key={manual.id} className="border-b">
                <td className="px-4 py-2 text-gray-700 text-center">{manual.name}</td>
                <td className="px-4 py-2 text-gray-700 text-center">{manual.added_at ? new Date(manual.added_at).toLocaleDateString() : '—'}</td>
                <td className="px-4 py-2 text-center">
                  <button
                    onClick={() => handleDeleteManual(manual.id)}
                    className="p-2 rounded-full text-red-500 hover:text-red-700 transition-colors inline-flex items-center justify-center"
                    title="Delete manual"
                  >
                    x
                  </button>
                </td>
              </tr>
            ))
          ) : (
            <tr>
               <td colSpan={3} className="px-4 py-10 text-black-500 italic text-center">
                  No hay manuales para mostrar.
               </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  </div>
  );
};

export default ManualTable;


