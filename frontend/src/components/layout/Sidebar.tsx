import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Activity, ShieldAlert, Settings } from 'lucide-react';
import { motion } from 'framer-motion';
import classes from './Sidebar.module.css';
import clsx from 'clsx';

const navItems = [
    { icon: LayoutDashboard, label: 'Overview', path: '/' },
    { icon: Activity, label: 'Real-time Metrics', path: '/metrics' },
    { icon: ShieldAlert, label: 'Security Events', path: '/events' },
    { icon: Settings, label: 'Settings', path: '/settings' },
];

export const Sidebar: React.FC = () => {
    return (
        <aside className={classes.sidebar}>
            <div className={classes.logo}>
                <div className={classes.logoIcon} />
                <span>PoisonGuard</span>
            </div>

            <nav className={classes.nav}>
                {navItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) => clsx(classes.link, isActive && classes.active)}
                    >
                        {({ isActive }) => (
                            <>
                                <item.icon size={20} />
                                <span>{item.label}</span>
                                {isActive && (
                                    <motion.div
                                        layoutId="active-indicator"
                                        className={classes.activeIndicator}
                                        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                                    />
                                )}
                            </>
                        )}
                    </NavLink>
                ))}
            </nav>

            <div className={classes.footer}>
                <div className={classes.statusDot} />
                <span>System Online</span>
            </div>
        </aside>
    );
};
