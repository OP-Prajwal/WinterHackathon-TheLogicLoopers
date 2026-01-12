import React from 'react';
import { Bell, User } from 'lucide-react';

export const Header: React.FC = () => {
    return (
        <header className="h-20 px-8 flex items-center justify-between z-10">
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
                <div className="flex items-center gap-3 pl-6 border-l border-white/5">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-violet-500 p-[1px]">
                        <div className="w-full h-full rounded-full bg-dark-900 flex items-center justify-center">
                            <User size={16} className="text-gray-300" />
                        </div>
                    </div>
                    <span className="text-sm font-medium text-gray-300">Admin</span>
                </div>
            </div>
        </header>
    );
};
