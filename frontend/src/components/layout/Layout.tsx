import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';

const Layout: React.FC = () => {
    return (
        <div className="flex min-h-screen bg-background text-foreground font-sans antialiased">
            <Sidebar />
            <div className="flex-1 flex flex-col pl-64">
                <Header />
                <main className="flex-1 pt-16 p-6 overflow-y-auto">
                    <Outlet />
                </main>
            </div>
        </div>
    );
};

export default Layout;
