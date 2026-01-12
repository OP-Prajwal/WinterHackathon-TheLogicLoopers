import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Activity, ShieldAlert, Settings } from 'lucide-react';
import { motion } from 'framer-motion';
import clsx from 'clsx';

const navItems = [
    { icon: LayoutDashboard, label: 'Overview', path: '/' },
    { icon: Activity, label: 'Personal Zone', path: '/metrics' },
    { icon: ShieldAlert, label: 'Security Events', path: '/events' },
    { icon: Settings, label: 'Settings', path: '/settings' },
];

export const Sidebar: React.FC = () => {
    return (
        <aside className="w-64 flex flex-col glass-panel m-4 mr-0 border-r-0 rounded-r-none relative z-20 overflow-hidden">
            <div className="flex items-center gap-3 p-6 mb-4 border-b border-white/5 relative">
                {/* Glowing Logo */}
                <div className="relative w-10 h-10 flex items-center justify-center bg-dark-800 rounded-lg border border-cyan-500/30 shadow-[0_0_15px_rgba(6,182,212,0.3)]">
                    <div className="w-6 h-6 bg-cyan-400 rounded-md rotate-45 transform" />
                    <div className="absolute inset-0 bg-cyan-400/20 blur-lg rounded-full" />
                </div>
                <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-transparent">
                    PoisonGuard
                </span>
            </div>

            <nav className="flex-1 px-4 py-2 space-y-1">
                {navItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        end={item.path === '/'} // Only match exact for root, prefix for others if needed
                        className={({ isActive }) => clsx(
                            "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 relative group overflow-hidden",
                            isActive
                                ? "bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 shadow-[0_0_15px_rgba(6,182,212,0.1)]"
                                : "text-gray-400 hover:text-gray-100 hover:bg-white/5"
                        )}
                    >
                        {({ isActive }) => (
                            <>
                                <item.icon size={20} className={clsx("transition-colors", isActive ? "text-cyan-400 drop-shadow-[0_0_8px_rgba(34,211,238,0.8)]" : "group-hover:text-white")} />
                                <span className={clsx("font-medium", isActive && "text-glow-cyan")}>{item.label}</span>

                                {/* Active Indicator Bar */}
                                {isActive && (
                                    <motion.div
                                        layoutId="active-nav"
                                        className="absolute left-0 top-0 bottom-0 w-1 bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,1)]"
                                    />
                                )}
                            </>
                        )}
                    </NavLink>
                ))}
            </nav>

            <div className="p-4 mt-auto">
                <div className="p-4 rounded-xl bg-gradient-to-br from-dark-800 to-dark-900 border border-white/5">
                    <div className="flex items-center gap-2 mb-2">
                        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]" />
                        <span className="text-xs font-mono text-emerald-400">SYSTEM ONLINE</span>
                    </div>
                </div>
            </div>
        </aside>
    );
};
