import React, { useState, useEffect } from 'react';
import { Shield, ShieldOff, RotateCcw, Lock, Unlock } from 'lucide-react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { API_BASE } from '../../services/api';

export const ControlPanel: React.FC = () => {
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
        <div className="glass-panel w-full p-6 relative overflow-hidden min-h-[180px] flex flex-col justify-center">
            <div className="flex items-center gap-2 mb-4">
                <Shield size={18} className="text-cyan-400" />
                <h3 className="text-lg font-bold text-gray-200">System Protocols</h3>
            </div>

            <div className="space-y-4">
                {/* Strict Mode Toggle */}
                <div className="flex items-center justify-between p-3 rounded-xl bg-dark-900/50 border border-white/5">
                    <div className="flex items-center gap-3">
                        <div className={clsx("p-2 rounded-lg", strictMode ? "bg-cyan-500/20 text-cyan-400" : "bg-gray-700/30 text-gray-500")}>
                            {strictMode ? <Lock size={18} /> : <Unlock size={18} />}
                        </div>
                        <div>
                            <div className="text-sm font-medium text-gray-300">Strict Mode</div>
                            <div className="text-xs text-gray-500">Auto-Halt on detected threats</div>
                        </div>
                    </div>
                    <motion.button
                        whileTap={{ scale: 0.95 }}
                        onClick={toggleStrictMode}
                        className={clsx(
                            "w-12 h-6 rounded-full relative transition-colors duration-300",
                            strictMode ? "bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.4)]" : "bg-gray-700"
                        )}
                    >
                        <motion.div
                            className="absolute top-1 bottom-1 bg-white rounded-full w-4 shadow-sm"
                            animate={{ x: strictMode ? 26 : 2 }}
                        />
                    </motion.button>
                </div>

                {/* Halt Override */}
                {isHalted && (
                    <motion.button
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={resetHalt}
                        className="w-full py-3 flex items-center justify-center gap-2 bg-red-500/20 border border-red-500/50 text-red-400 rounded-xl hover:bg-red-500/30 transition-all font-bold tracking-wide shadow-[0_0_15px_rgba(239,68,68,0.2)] animate-pulse"
                    >
                        <RotateCcw size={18} />
                        SYSTEM HALTED - CLICK TO RESET
                    </motion.button>
                )}
            </div>
        </div>
    );
};
