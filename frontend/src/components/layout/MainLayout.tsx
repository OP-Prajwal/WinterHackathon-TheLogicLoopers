import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { Outlet } from 'react-router-dom';
import classes from './MainLayout.module.css';

export const MainLayout: React.FC = () => {
    return (
        <div className={classes.container}>
            <Sidebar />
            <div className={classes.contentWrapper}>
                <Header />
                <main className={classes.main}>
                    <Outlet />
                </main>
            </div>
        </div>
    );
};
