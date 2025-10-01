"use client";
import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const Navbar = () => {
  const pathname = usePathname() || '/';

  return (
    <div className="bg-white shadow px-4 py-3 flex justify-between items-center">
      <Link 
        href="/"
        className="text-lg font-bold text-gray-900 hover:text-gray-700 transition-colors"
      >
        Maintenance Assistant
      </Link>

      <div className="flex items-center gap-4">
        {pathname === '/' ? (
          <>
            <Link 
              href="/manuals" 
              className="text-base font-normal text-gray-900 px-3 py-2 rounded-2xl hover:bg-gray-100 transition-colors cursor-pointer"
            >
              Manuals
            </Link>
          </>
        ) : (
          <Link 
            href="/" 
            className="text-base font-normal text-gray-900 px-3 py-2 rounded-2xl hover:bg-gray-100 transition-colors cursor-pointer"
          >
            Chat
          </Link>
        )}

        <button className="flex items-center gap-3 px-3 py-2 rounded-2xl hover:bg-gray-100 transition-colors cursor-pointer">
          <span>Samuel Jauregui</span>
          <div className="w-8 h-8 rounded-full bg-gray-300 border-2 border-gray-400"></div>
        </button>
      </div>
    </div>
  );
};

export default Navbar;

