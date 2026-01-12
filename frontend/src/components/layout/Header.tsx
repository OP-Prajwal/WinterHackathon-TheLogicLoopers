import React, { useState, useRef, useEffect } from 'react';
import { Bell, User, LogOut, Database, ChevronDown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export const Header: React.FC = () => {
    const [isProfileOpen, setIsProfileOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);
    const navigate = useNavigate();
    const username = localStorage.getItem('username') || 'Admin';

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsProfileOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleLogout = () => {
        // Clear tokens/state if needed
        // localStorage.removeItem('token');
        navigate('/login'); // Redirect to login (assuming route exists, or just refresh)
    };

    return (
        <header className="h-20 px-8 flex items-center justify-between z-50 relative">
            <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-100 to-gray-400 bg-clip-text text-transparent">
                    Dashboard
                </h1>
                <span className="text-sm text-gray-500 font-mono tracking-wide">
                    Monitoring BRFSS Diabetes Training
                </span>
            </div>

            <div className="flex items-center gap-6">
                <button className="relative p-2 rounded-lg text-gray-400 hover:text-cyan-400 hover:bg-white/5 transition-all">
                    <Bell size={20} />
                    <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-rose-500 rounded-full shadow-[0_0_8px_rgba(244,63,94,0.6)] animate-pulse" />
                </button>

                <div className="relative" ref={dropdownRef}>
                    <button
                        onClick={() => setIsProfileOpen(!isProfileOpen)}
                        className={`flex items-center gap-3 pl-6 border-l border-white/5 transition-all group ${isProfileOpen ? 'opacity-100' : 'opacity-80 hover:opacity-100'}`}
                    >
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-violet-500 p-[1px]">
                            <div className="w-full h-full rounded-full bg-dark-900 flex items-center justify-center">
                                <User size={16} className="text-gray-300" />
                            </div>
                        </div>
                        <span className="text-sm font-medium text-gray-300">{username}</span>
                        <ChevronDown size={14} className={`text-gray-500 transition-transform duration-300 ${isProfileOpen ? 'rotate-180' : ''}`} />
                    </button>

                    {/* Dropdown Menu */}
                    {isProfileOpen && (
                        <div className="absolute right-0 top-full mt-2 w-64 bg-dark-800 border border-white/10 rounded-xl shadow-[0_10px_40px_rgba(0,0,0,0.5)] backdrop-blur-xl overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200">
                            {/* Profile Header */}
                            <div className="p-4 border-b border-white/5 bg-white/5">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-violet-500 p-[1px]">
                                        <div className="w-full h-full rounded-full bg-dark-900 flex items-center justify-center">
                                            <User size={20} className="text-gray-200" />
                                        </div>
                                    </div>
                                    <div>
                                        <h4 className="text-sm font-bold text-white">{username}</h4>
                                        <p className="text-xs text-gray-400">User</p>
                                    </div>
                                </div>
                            </div>

                            {/* Menu Items */}
                            <div className="p-2 space-y-1">
                                <div className="px-3 py-2">
                                    <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">My Datasets</p>
                                    <button
                                        onClick={() => {
                                            setIsProfileOpen(false);
                                            navigate('/datasets');
                                        }}
                                        className="w-full flex items-center gap-3 p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 hover:bg-cyan-500/20 transition-colors text-left"
                                    >
                                        <Database size={16} />
                                        <span className="text-sm font-medium">My Datasets</span>
                                        <div className="ml-auto w-2 h-2 bg-cyan-400 rounded-full shadow-[0_0_8px_rgba(34,211,238,0.8)] animate-pulse" />
                                    </button>
                                </div>

                                <div className="h-px bg-white/5 my-1" />

                                <button
                                    onClick={handleLogout}
                                    className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-rose-400 hover:bg-rose-500/10 transition-colors text-sm font-medium"
                                >
                                    <LogOut size={16} />
                                    Sign Out
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </header>
    );
};
