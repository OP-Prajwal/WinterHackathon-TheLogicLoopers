import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { Outlet } from 'react-router-dom';

export const MainLayout: React.FC = () => {
    return (
        <div className="flex h-screen w-full bg-dark-900 bg-cyber-grid text-gray-200 overflow-hidden relative">
            {/* Background Gradient Overlay */}
            <div className="absolute inset-0 pointer-events-none bg-gradient-radial from-transparent via-dark-900/50 to-dark-900 z-0 opacity-80" />

            <div className="relative z-10 flex w-full h-full">
                <Sidebar />
                <div className="flex flex-col flex-1 h-full min-w-0 overflow-hidden">
                    <Header />
                    <main className="flex-1 overflow-y-auto p-4 md:p-6 scrollbar-thin scrollbar-thumb-dark-700 scrollbar-track-transparent">
                        <Outlet />
                    </main>
                </div>
            </div>
        </div>
    );
};
