import React from 'react';
import { Bell, User } from 'lucide-react';
import classes from './Header.module.css';

export const Header: React.FC = () => {
    return (
        <header className={classes.header}>
            <div className={classes.title}>
                <h1>Dashboard</h1>
                <span className={classes.subtitle}>Monitoring BRFSS Diabetes Training</span>
            </div>

            <div className={classes.actions}>
                <button className={classes.iconBtn}>
                    <Bell size={20} />
                    <span className={classes.badge}>2</span>
                </button>
                <div className={classes.profile}>
                    <div className={classes.avatar}>
                        <User size={18} />
                    </div>
                    <span>Admin</span>
                </div>
            </div>
        </header>
    );
};
