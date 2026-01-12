import React from 'react';

const Header: React.FC = () => {
    return (
        <header className="h-16 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 fixed top-0 w-full z-10 pl-64">
            <div className="flex h-14 items-center px-6">
                <h2 className="font-semibold text-lg">Dashboard</h2>
                <div className="ml-auto flex items-center space-x-4">
                    {/* Add header actions here if needed */}
                </div>
            </div>
        </header>
    );
};

export default Header;
