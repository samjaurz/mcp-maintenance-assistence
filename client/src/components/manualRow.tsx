import React from 'react';
import { Manual } from '@/types/manuals';

const manualRow = ({ manual, onDelete }: { manual: Manual, onDelete: (id: number) => void }) => {
  return (
    <tr className="bg-white border-b border-gray-200 hover:bg-gray-50 w-full">
      <td className="px-4 py-3 whitespace-nowrap text-center w-1/4">
        <div className="flex flex-col items-start gap-1 px-2">
          <span className="font-semibold text-gray-800">{manual.name}</span>
        </div>
      </td>
      <td className="px-4 py-3 text-gray-600 whitespace-nowrap text-center w-1/6">
        {manual.added_at ? new Date(manual.added_at).toLocaleDateString() : '—'}
      </td>
      <td className="px-4 py-3 text-gray-400 w-1/6 text-center">
        <div className="flex items-center justify-center gap-4">
          <button className="text-gray-400 hover:text-gray-600 transition-colors"
           onClick={() => onDelete(manual.id)}>
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className="h-4 w-4" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="red"  
            strokeWidth="2" 
            strokeLinecap="round" 
            strokeLinejoin="round"
          >
          <path d="M18 6 L6 18 M6 6 L18 18"/>
          </svg>
          </button>
        </div>
      </td>
    </tr>
  );
};

export default manualRow;