// components/Layout.tsx
import Link from 'next/link';
import React, { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-blue-600 text-white p-4">
        <nav className="container mx-auto">
          <ul className="flex space-x-4">
            <li><Link href="/" className="hover:underline">Home</Link></li>
            <li><Link href="/unit1" className="hover:underline">Unit 1: Digital Information</Link></li>
            {/* Add more units as they become available */}
          </ul>
        </nav>
      </header>
      <main className="flex-grow container mx-auto px-4 py-8">
        {children}
      </main>
      <footer className="bg-gray-200 p-4 text-center">
        {/* <p>&copy; {new Date().getFullYear()} Digital Information Curriculum. All rights reserved.</p> */}
      </footer>
    </div>
  );
};

export default Layout;