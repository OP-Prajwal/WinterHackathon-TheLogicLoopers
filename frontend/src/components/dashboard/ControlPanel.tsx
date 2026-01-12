import React, { useState, useEffect } from 'react';
import { Play, Square, Skull, Shield, ShieldOff, RotateCcw } from 'lucide-react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import classes from './ControlPanel.module.css';
import { API_BASE } from '../../services/api';

interface ControlPanelProps {
    isRunning: boolean;
    isPoisoned: boolean;
    onStart: () => void;
    onStop: () => void;
    onInject: () => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
    isRunning,
    isPoisoned,
    onStart,
    onStop,
    onInject
}) => {
    const [strictMode, setStrictMode] = useState(true);
    const [isHalted, setIsHalted] = useState(false);

    // Fetch current settings on mount
    useEffect(() => {
        fetch(`${API_BASE}/api/settings`)
            .then(res => res.json())
            .then(data => {
                setStrictMode(data.strict_mode);
                setIsHalted(data.halted);
            })
            .catch(console.error);
    }, []);

    const toggleStrictMode = async () => {
        const newValue = !strictMode;
        try {
            const res = await fetch(`${API_BASE}/api/settings/strict-mode?enabled=${newValue}`, {
                method: 'POST'
            });
            const data = await res.json();
            setStrictMode(data.strict_mode);
        } catch (error) {
            console.error('Failed to toggle strict mode:', error);
        }
    };

    const resetHalt = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/settings/reset-halt`, { method: 'POST' });
            const data = await res.json();
            setIsHalted(data.halted);
        } catch (error) {
            console.error('Failed to reset halt:', error);
        }
    };

    return (
        <div className={classes.panel}>
            <h3 className={classes.heading}>Controls</h3>

            {/* Strict Mode Toggle */}
            <div className={classes.strictModeRow}>
                <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className={clsx(classes.btn, classes.strictToggle, strictMode && classes.strictActive)}
                    onClick={toggleStrictMode}
                >
                    {strictMode ? <Shield size={18} /> : <ShieldOff size={18} />}
                    Strict Mode: {strictMode ? 'ON' : 'OFF'}
                </motion.button>

                {isHalted && (
                    <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className={clsx(classes.btn, classes.reset)}
                        onClick={resetHalt}
                    >
                        <RotateCcw size={18} />
                        Reset HALT
                    </motion.button>
                )}
            </div>

            <div className={classes.grid}>
                {!isRunning ? (
                    <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className={clsx(classes.btn, classes.start)}
                        onClick={onStart}
                        disabled={isHalted}
                    >
                        <Play size={20} fill="currentColor" />
                        {isHalted ? 'System HALTED' : 'Start Monitoring'}
                    </motion.button>
                ) : (
                    <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className={clsx(classes.btn, classes.stop)}
                        onClick={onStop}
                    >
                        <Square size={20} fill="currentColor" />
                        Stop
                    </motion.button>
                )}

                <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className={clsx(classes.btn, classes.poison)}
                    onClick={onInject}
                    disabled={!isRunning || isPoisoned}
                >
                    <Skull size={20} />
                    {isPoisoned ? 'Attack Active' : 'Simulate Attack'}
                </motion.button>
            </div>
        </div>
    );
};
