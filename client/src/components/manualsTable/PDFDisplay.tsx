import React, { useState, useMemo, useRef } from "react";
import axiosInstance from "@/services/axiosInstance";

interface PageInfo {
  page_number: number;
  extracted_text: string;
  has_error: boolean;
  images: boolean;
}

interface InitialData {
  pages_total: number;
  pdf_filename: string;
  pages_info: PageInfo[];
}

interface Selection {
  x: number;
  y: number;
  width: number;
  height: number;
}

const PDFDisplay: React.FC = () => {
  const [initialData, setInitialData] = useState<InitialData | null>(null);
  const [pagesData, setPagesData] = useState<PageInfo[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [isPaused, setIsPaused] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isFinished, setIsFinished] = useState(false);
  const [imageUrlToReview, setImageUrlToReview] = useState<string | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  
  // 🔥 NUEVOS ESTADOS: Para la selección de imagen
  const [selection, setSelection] = useState<Selection | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const imageRef = useRef<HTMLImageElement>(null);

  const handleStartProcessing = async () => {
    try {
      setIsProcessing(true);
      setIsFinished(false);
      setIsPaused(false);
      setImageUrlToReview(null);
      setSelection(null); // 🔥 NUEVO: Limpiar selección al iniciar

      const res = await axiosInstance.get(`/manuals/get_manual`);
      console.log("Respuesta de /manualsTable/get_manual:", res.data);

      setInitialData(res.data);
      setPagesData(res.data.pages_info);
      setCurrentPage(1);
      setSourceUrl(res.data.source_url);

      await processNextPage(1);
    } catch (error) {
      console.error("Error iniciando procesamiento:", error);
      setIsProcessing(false);
    }
  };

  const processNextPage = async (page: number) => {
    if (!initialData) return;

    try {
      // Buscar info de la página actual
      const currentPageInfo = initialData.pages_info.find(
        (p) => p.page_number === page
      );

      if (!currentPageInfo) {
        console.error(`No se encontró información para la página ${page}`);
        return;
      }

      // Enviar la página al backend
      const res = await axiosInstance.post(
        `/manuals/process_page/`,
        currentPageInfo
      );

      console.log(`✅ Página ${page} procesada:`, res.data);

      // ✅ CORRECCIÓN: Verificar el has_error del objeto original, no de la respuesta
      if (currentPageInfo.has_error) {
        setCurrentPage(page);
        await handlerErrorPage(page, initialData.pdf_filename);
        return;
      }

      if (page < initialData.pages_total) {
        setCurrentPage(page + 1);
        await processNextPage(page + 1);
      } else {
        setIsFinished(true);
        setIsProcessing(false);
      }
    } catch (err) {
      console.error("Error procesando página:", err);
      setIsPaused(true);
      setIsProcessing(false);
    }
  };

  const handlerErrorPage = async (page: number, filename: string) => {
    try {
      // Verifica la estructura exacta que espera tu PageInfoRequest
      const requestData = {
        page_number: page,
        pdf_filename: sourceUrl, // o el nombre exacto del campo que espera tu backend
      };

      console.log("Enviando a retrieve_page:", requestData);

      const res = await axiosInstance.post(`/manuals/retrieve_page/`, requestData);
      console.log("🔎 Respuesta de retrieve_page:", res.data);

      // El backend devuelve { "url": "..." }
      if (res.data.url) {
        setImageUrlToReview(res.data.url);
        setIsPaused(true);
        setIsProcessing(false);
      } else {
        console.error("No se recibió URL en la respuesta");
        setIsPaused(true);
        setIsProcessing(false);
      }
    } catch (error) {
      console.error("Error recuperando página:", error);
      setImageUrlToReview(null);
      setIsPaused(true);
      setIsProcessing(false);
    }
  };

  // 🔥 NUEVA FUNCIÓN: Manejar inicio de selección
  const handleImageMouseDown = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imageRef.current) return;
    
    const rect = imageRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setIsSelecting(true);
    setStartPos({ x, y });
    setSelection({ x, y, width: 0, height: 0 });
  };

  // 🔥 NUEVA FUNCIÓN: Manejar movimiento durante selección
  const handleImageMouseMove = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!isSelecting || !imageRef.current) return;
    
    const rect = imageRef.current.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    
    const width = currentX - startPos.x;
    const height = currentY - startPos.y;
    
    setSelection({
      x: width < 0 ? currentX : startPos.x,
      y: height < 0 ? currentY : startPos.y,
      width: Math.abs(width),
      height: Math.abs(height)
    });
  };

  // 🔥 NUEVA FUNCIÓN: Manejar fin de selección
  const handleImageMouseUp = () => {
    setIsSelecting(false);
  };

  // 🔥 NUEVA FUNCIÓN: Enviar selección al backend
  const sendSelectionToBackend = async () => {
    if (!selection || !initialData || !imageUrlToReview) return;

    try {
      const selectionData = {
        page_number: currentPage,
        pdf_filename: initialData.pdf_filename,
        selection: {
          x: selection.x,
          y: selection.y,
          width: selection.width,
          height: selection.height
        },
        image_url: imageUrlToReview
      };

      console.log("📤 Enviando selección al backend:", selectionData);
      
      // Enviar al backend
      const res = await axiosInstance.post(`/manuals/process_selection/`, selectionData);
      console.log("✅ Selección procesada:", res.data);
      
      setSelection(null);
      
    } catch (error) {
      console.error("Error enviando selección:", error);
    }
  };

  const handleUserConfirmation = async () => {
    if (!initialData) return;

    // Marcar la página como corregida
    setPagesData((prevData) =>
      prevData.map((p) =>
        p.page_number === currentPage ? { ...p, has_error: false } : p
      )
    );

    setImageUrlToReview(null);
    setSelection(null); // 🔥 NUEVO: Limpiar selección al continuar
    setIsPaused(false);
    setIsProcessing(true);

    // Continuar con la siguiente
    await processNextPage(currentPage + 1);
  };

  const currentStatus = useMemo(() => {
    if (isFinished) return "✅ Proceso Finalizado.";
    if (isPaused) return "⚠️ Pausado por Error.";
    if (isProcessing)
      return `⚙️ Procesando página ${currentPage} de ${
        initialData?.pages_total ?? 0
      }...`;
    return "Click en 'Iniciar procesamiento'.";
  }, [isFinished, isPaused, isProcessing, currentPage, initialData]);

  return (
    <div className="flex-1 bg-white rounded-2xl shadow-xl border border-gray-100 flex flex-col p-6">
      <div className="flex justify-between items-center pb-4 border-b border-gray-200 mb-4">
        <h3 className="text-2xl font-bold text-gray-800">
          Revisor de Manuales PDF
        </h3>
        <span
          className={`px-4 py-2 text-sm font-semibold rounded-full ${
            isFinished
              ? "bg-green-100 text-green-700"
              : isPaused
              ? "bg-yellow-100 text-yellow-700"
              : isProcessing
              ? "bg-blue-100 text-blue-700"
              : "bg-gray-100 text-gray-700"
          }`}
        >
          {currentStatus}
        </span>
      </div>

      <div className="flex flex-col items-center justify-center gap-4 p-4 border border-gray-200 rounded-lg bg-gray-50">
        <h4 className="text-lg font-semibold text-gray-700">Controles</h4>
        <div className="flex gap-4 flex-wrap justify-center">
          <button onClick={handleStartProcessing}>
            Iniciar procesamiento
          </button>
          <button onClick={handleUserConfirmation}>
            Corregir y Continuar
          </button>
          <button
            onClick={() => {
              setCurrentPage(1);
              setIsProcessing(false);
              setIsPaused(false);
              setIsFinished(false);
              setInitialData(null);
              setPagesData([]);
              setImageUrlToReview(null);
              setSelection(null); // 🔥 NUEVO: Limpiar selección al reiniciar
            }}
          >
            Reiniciar
          </button>
        </div>
      </div>

      {initialData && isPaused && (
        <div className="overflow-auto p-4 bg-yellow-50 border-yellow-300 border rounded-xl space-y-4 shadow-inner mt-4">
          <h2 className="text-xl font-bold text-yellow-800">
            ⚠️ Revisión de Página {currentPage} ({initialData.pdf_filename})
          </h2>
          <div className="bg-white p-4 rounded-lg shadow-md border border-gray-100">
            {imageUrlToReview && (
              <div style={{position: 'relative', display: 'inline-block'}}>
                <img
                  ref={imageRef}
                  src={imageUrlToReview}
                  alt={`Página ${currentPage}`}
                  draggable={false} 
                  style={{
                    maxWidth: '100%',
                    height: 'auto',
                    border: '2px solid #d1d5db',
                    borderRadius: '0.5rem',
                    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
                    cursor: 'crosshair'
                  }}
                  onMouseDown={handleImageMouseDown}
                  onMouseMove={handleImageMouseMove}
                  onMouseUp={handleImageMouseUp}
                  onMouseLeave={handleImageMouseUp}
                />
                {selection && (
                  <div
                    style={{
                      position: 'absolute',
                      border: '2px solid #ef4444',
                      backgroundColor: 'rgba(239, 68, 68, 0.3)',
                      pointerEvents: 'none',
                      left: `${selection.x}px`,
                      top: `${selection.y}px`,
                      width: `${selection.width}px`,
                      height: `${selection.height}px`
                    }}
                  />
                )}
              </div>
            )}
            
            {/* 🔥 NUEVO: Botones para manejar selección */}
            {selection && (
              <div style={{marginTop: '1rem', display: 'flex', gap: '1rem', justifyContent: 'center'}}>
                <button
                  onClick={sendSelectionToBackend}
                  style={{
                    padding: '0.5rem 1.5rem',
                    backgroundColor: '#9333ea',
                    color: 'white',
                    fontWeight: '600',
                    borderRadius: '0.5rem',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    border: 'none',
                    cursor: 'pointer'
                  }}
                >
                  📤 Enviar Selección al Backend
                </button>
                <button
                  onClick={() => setSelection(null)}
                  style={{
                    padding: '0.5rem 1.5rem',
                    backgroundColor: '#6b7280',
                    color: 'white',
                    fontWeight: '600',
                    borderRadius: '0.5rem',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    border: 'none',
                    cursor: 'pointer'
                  }}
                >
                  ❌ Limpiar Selección
                </button>
              </div>
            )}
            
            {/* 🔥 NUEVO: Instrucciones para el usuario */}
            {!selection && (
              <div style={{marginTop: '1rem', textAlign: 'center', fontSize: '0.875rem', color: '#6b7280'}}>
                <p>🖱️ <strong>Para seleccionar un área:</strong> Haz clic y arrastra sobre la imagen</p>
              </div>
            )}
          </div>
        </div>
      )}

      {initialData && isProcessing && (
        <div className="flex items-center justify-center text-center text-gray-500 mt-4 p-4 bg-blue-50 rounded-lg">
          <p className="text-lg font-semibold">
            Procesando... Página {currentPage} de {initialData.pages_total}
          </p>
        </div>
      )}

      {isFinished && (
        <div className="flex items-center justify-center text-center p-4 bg-green-50 rounded-lg mt-4">
          <p className="text-xl font-bold text-green-700">
            ¡Proceso Terminado! 🎉
          </p>
        </div>
      )}
    </div>
  );
};

export default PDFDisplay;