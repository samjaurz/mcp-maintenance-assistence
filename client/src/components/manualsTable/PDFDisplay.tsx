"use client";

export default function PDFDisplay() {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Vista Previa de PDF</h2>

      <div className="w-full h-[600px] border border-gray-300 rounded flex items-center justify-center bg-gray-50">
        <div className="text-center text-gray-500">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <p className="mt-2">Selecciona un manual para ver su contenido</p>
        </div>
      </div>
    </div>
  );
}

