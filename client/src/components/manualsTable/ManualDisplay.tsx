"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import { Manual } from "@/types/manuals";

export default function ManualDisplay() {
  const [manuals, setManuals] = useState<Manual[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    fetchManuals();
  }, []);

  const fetchManuals = async () => {
    try {
      setLoading(true);
      const response = await axios.get("http://localhost:8000/manuals");
      setManuals(response.data);
      setError(null);
    } catch (err) {
      setError("Error al cargar los manuales");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      await axios.post("http://localhost:8000/manuals/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      await fetchManuals();
      event.target.value = "";
    } catch (err) {
      setError("Error al subir el archivo");
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (manualId: number) => {
    if (!confirm("¿Estás seguro de eliminar este manual?")) return;

    try {
      await axios.delete(`http://localhost:8000/manuals/${manualId}`);
      await fetchManuals();
    } catch (err) {
      setError("Error al eliminar el manual");
      console.error(err);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-3">
            <div className="h-4 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold text-gray-800">Manuales</h2>
        <div>
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            className="hidden"
            id="pdf-upload"
            disabled={uploading}
          />
          <label
            htmlFor="pdf-upload"
            className={`px-4 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 transition ${
              uploading ? "opacity-50 cursor-not-allowed" : ""
            }`}
          >
            {uploading ? "Subiendo..." : "Subir PDF"}
          </label>
        </div>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                ID
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Nombre
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Fecha de Carga
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Acciones
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {manuals.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-6 py-4 text-center text-gray-500">
                  No hay manuales cargados
                </td>
              </tr>
            ) : (
              manuals.map((manual) => (
                <tr key={manual.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {manual.id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {manual.name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(manual.added_at).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <button
                      onClick={() => handleDelete(manual.id)}
                      className="text-red-600 hover:text-red-900 transition"
                    >
                      Eliminar
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

