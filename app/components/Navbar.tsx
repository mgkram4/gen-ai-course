'use client';

import { Bars3Icon, BeakerIcon, BookOpenIcon, ChevronDownIcon, CodeBracketIcon, DocumentTextIcon, XMarkIcon } from '@heroicons/react/24/solid';
import { AnimatePresence, motion } from 'framer-motion';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';

interface NavbarProps {
  className?: string;
}

const ModernNavbar = ({ className }: NavbarProps) => {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState<number | null>(null);
  const dropdownRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [scrolled, setScrolled] = useState(false);

  // Handle scrolling effect
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!dropdownRefs.current.some(ref => ref?.contains(event.target as Node))) {
        setActiveDropdown(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const units = [
    { id: 1, title: 'Foundations of Generative AI', hasExercises: true, hasTerminology: true },
    { id: 2, title: 'Working with Language Models', hasExercises: true, hasTerminology: true },
    { id: 3, title: 'Advanced LLM Concepts', hasExercises: true, hasTerminology: true },
    { id: 4, title: 'Image Generation & Computer Vision', hasExercises: true, hasTerminology: true },
    { id: 5, title: 'Audio & Speech AI', hasExercises: true, hasTerminology: true },
    { id: 6, title: 'AI Agents & Automation', hasExercises: true, hasTerminology: true },
    { id: 7, title: 'Deployment & Production', hasExercises: true, hasTerminology: true },
    { id: 8, title: 'AI Ethics & Safety', hasExercises: true, hasTerminology: true },
    { id: 9, title: 'Business Applications', hasExercises: true, hasTerminology: true },
  ];
  // Improved dropdown menu component
  const DropdownMenu = ({ unit, isActive }: { unit: { id: number; title: string; hasTerminology: boolean; hasExercises: boolean }; isActive: boolean }) => (
    <AnimatePresence>
      {isActive && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          transition={{ duration: 0.2 }}
          className="absolute top-full left-0 mt-2 w-60 bg-white rounded-lg shadow-xl border border-gray-100 overflow-hidden z-50"
        >
          <div className="p-2">
            <Link
              href={`/unit${unit.id}`}
              className="flex items-center px-4 py-2 text-gray-700 hover:bg-blue-50 rounded-md transition-colors"
            >
              <BookOpenIcon className="h-5 w-5 mr-2 text-blue-500" />
              <span>Overview</span>
            </Link>
            {unit.hasExercises && (
              <Link
                href={`/unit${unit.id}/exercises`}
                className="flex items-center px-4 py-2 text-gray-700 hover:bg-blue-50 rounded-md transition-colors"
              >
                <BeakerIcon className="h-5 w-5 mr-2 text-blue-500" />
                <span>Exercises</span>
              </Link>
            )}
            {unit.hasTerminology && (
              <Link
                href={`/unit${unit.id}/terminology`}
                className="flex items-center px-4 py-2 text-gray-700 hover:bg-blue-50 rounded-md transition-colors"
              >
                <DocumentTextIcon className="h-5 w-5 mr-2 text-blue-500" />
                <span>Terminology</span>
              </Link>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  return (
    <>
      <nav className={`fixed w-full z-50 transition-all duration-300 ${
        scrolled ? 'bg-white shadow-md' : 'bg-white'
      } ${className || ''}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link href="/" className="flex items-center space-x-2">
              <CodeBracketIcon className="h-8 w-8 text-blue-600" />
              <span className="text-gray-900 font-bold text-xl">GenAI Learning</span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden lg:flex items-center space-x-2">
              <div className="flex items-center space-x-2">
                {units.map((unit) => (
                  <div 
                    key={unit.id} 
                    className="relative"
                    ref={el => { dropdownRefs.current[unit.id] = el }}
                  >
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setActiveDropdown(activeDropdown === unit.id ? null : unit.id);
                      }}
                      className={`flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200
                        ${pathname.startsWith(`/unit${unit.id}`) 
                          ? 'bg-blue-100 text-blue-700' 
                          : 'text-gray-700 hover:bg-gray-100'}`}
                    >
                      <span>Unit {unit.id}</span>
                      <ChevronDownIcon className={`h-4 w-4 ml-1 transition-transform duration-200 
                        ${activeDropdown === unit.id ? 'transform rotate-180' : ''}`} />
                    </button>
                    <DropdownMenu unit={unit} isActive={activeDropdown === unit.id} />
                  </div>
                ))}
                <Link
                  href="/projects/chatgpt-clone"
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                    ${pathname === '/projects/chatgpt-clone' 
                      ? 'bg-green-100 text-green-700' 
                      : 'text-gray-700 hover:bg-gray-100'}`}
                >
                  ChatGPT Clone
                </Link>
                <Link
                  href="/projects/document-qa"
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                    ${pathname === '/projects/document-qa' 
                      ? 'bg-green-100 text-green-700' 
                      : 'text-gray-700 hover:bg-gray-100'}`}
                >
                  Document QA
                </Link>
                <Link
                  href="/projects/image-studio"
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                    ${pathname === '/projects/image-studio' 
                      ? 'bg-green-100 text-green-700' 
                      : 'text-gray-700 hover:bg-gray-100'}`}
                >
                  Image Studio
                </Link>
              </div>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 rounded-lg text-gray-600 hover:bg-gray-100 focus:outline-none"
            >
              {mobileMenuOpen ? (
                <XMarkIcon className="h-6 w-6" />
              ) : (
                <Bars3Icon className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="lg:hidden bg-white border-t"
            >
              <div className="max-h-[80vh] overflow-y-auto">
                <div className="px-4 py-2 space-y-1">
                  {units.map((unit) => (
                    <div key={unit.id} className="space-y-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setActiveDropdown(activeDropdown === unit.id ? null : unit.id);
                        }}
                        className={`w-full flex items-center justify-between p-3 rounded-lg text-left
                          ${pathname.startsWith(`/unit${unit.id}`) 
                            ? 'bg-blue-100 text-blue-700' 
                            : 'text-gray-700 hover:bg-gray-50'}`}
                      >
                        <span>Unit {unit.id}: {unit.title}</span>
                        <ChevronDownIcon className={`h-5 w-5 transition-transform duration-200 
                          ${activeDropdown === unit.id ? 'transform rotate-180' : ''}`} />
                      </button>
                      <AnimatePresence>
                        {activeDropdown === unit.id && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="pl-4 space-y-1"
                          >
                            <Link
                              href={`/unit${unit.id}`}
                              className="flex items-center p-3 rounded-lg text-gray-600 hover:bg-gray-50"
                            >
                              <BookOpenIcon className="h-5 w-5 mr-2 text-blue-500" />
                              Overview
                            </Link>
                            {unit.hasExercises && (
                              <Link
                                href={`/unit${unit.id}/exercises`}
                                className="flex items-center p-3 rounded-lg text-gray-600 hover:bg-gray-50"
                              >
                                <BeakerIcon className="h-5 w-5 mr-2 text-blue-500" />
                                Exercises
                              </Link>
                            )}
                            {unit.hasTerminology && (
                              <Link
                                href={`/unit${unit.id}/terminology`}
                                className="flex items-center p-3 rounded-lg text-gray-600 hover:bg-gray-50"
                              >
                                <DocumentTextIcon className="h-5 w-5 mr-2 text-blue-500" />
                                Terminology
                              </Link>
                            )}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  ))}
                  <Link
                    href="/projects/chatgpt-clone"
                    className={`block p-3 rounded-lg
                      ${pathname === '/projects/chatgpt-clone' 
                        ? 'bg-green-100 text-green-700' 
                        : 'text-gray-700 hover:bg-gray-50'}`}
                  >
                    ChatGPT Clone
                  </Link>
                  <Link
                    href="/projects/document-qa"
                    className={`block p-3 rounded-lg
                      ${pathname === '/projects/document-qa' 
                        ? 'bg-green-100 text-green-700' 
                        : 'text-gray-700 hover:bg-gray-50'}`}
                  >
                    Document QA
                  </Link>
                  <Link
                    href="/projects/image-studio"
                    className={`block p-3 rounded-lg
                      ${pathname === '/projects/image-studio' 
                        ? 'bg-green-100 text-green-700' 
                        : 'text-gray-700 hover:bg-gray-50'}`}
                  >
                    Image Studio
                  </Link>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </nav>
      <div className="h-16" />
    </>
  );
};

export default ModernNavbar;