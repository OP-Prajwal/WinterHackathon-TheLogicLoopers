import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, LayoutDashboard, Settings, User } from 'lucide-react';
import clsx from 'clsx';

const Sidebar: React.FC = () => {
    const navItems = [
        { icon: Home, label: 'Home', path: '/' },
        { icon: LayoutDashboard, label: 'Dashboard', path: '/dashboard' },
        { icon: User, label: 'Profile', path: '/profile' },
        { icon: Settings, label: 'Settings', path: '/settings' },
    ];

    return (
        <aside className="w-64 bg-card border-r border-border h-screen flex flex-col p-4 fixed left-0 top-0">
            <div className="flex items-center gap-2 mb-8 px-2">
                <div className="w-8 h-8 bg-primary rounded-lg"></div>
                <span className="font-bold text-xl">LogicLoopers</span>
            </div>

            <nav className="flex-1 space-y-1">
                {navItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) =>
                            clsx(
                                'flex items-center gap-3 px-3 py-2 rounded-md transition-colors',
                                isActive
                                    ? 'bg-primary text-primary-foreground'
                                    : 'hover:bg-accent hover:text-accent-foreground text-muted-foreground'
                            )
                        }
                    >
                        <item.icon size={20} />
                        <span className="font-medium">{item.label}</span>
                    </NavLink>
                ))}
            </nav>
        </aside>
    );
};

export default Sidebar;
