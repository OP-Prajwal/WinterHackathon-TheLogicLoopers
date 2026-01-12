import React, { useState, useEffect } from 'react';
import { Terminal, Cpu, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface SecurityAdvisorProps {
    metrics: {
        effective_rank: number;
        drift_score: number;
        batch: number;
    } | null;
}

const MESSAGES = {
    SAFE: [
        "System stability within nominal parameters.",
        "Representation alignment verified. No drift detected.",
        "Monitoring latent space for anomalies...",
        "Neural patterns match established baseline."
    ],
    WARNING: [
        "Minor representation drift detected. Monitoring closely.",
        "Increased feature density in recent batches. Investigating...",
        "Slight variance in model confidence scores.",
        "Scanning for subtle poisoning patterns..."
    ],
    DANGER: [
        "CRITICAL: Significant representation drift detected!",
        "UNAUTHORIZED PATTERN: Latent space corruption likely.",
        "ALERT: Model confidence dropping below safety threshold.",
        "IMMEDIATE ACTION REQUIRED: Purifying incoming stream."
    ]
};

export const SecurityAdvisor: React.FC<SecurityAdvisorProps> = ({ metrics }) => {
    const [currentText, setCurrentText] = useState("Initializing Neural Advisor...");
    const [status, setStatus] = useState<'safe' | 'warning' | 'danger'>('safe');

    useEffect(() => {
        if (!metrics) return;

        let level: 'safe' | 'warning' | 'danger' = 'safe';
        if (metrics.drift_score > 0.8) level = 'danger';
        else if (metrics.drift_score > 0.4) level = 'warning';

        setStatus(level);

        // Pick a random message for the current level
        const list = level === 'danger' ? MESSAGES.DANGER : level === 'warning' ? MESSAGES.WARNING : MESSAGES.SAFE;
        const msg = list[Math.floor(Math.random() * list.length)];

        // Only update if it's been a few seconds or level changed
        setCurrentText(msg);
    }, [metrics?.batch, metrics?.drift_score]);

    return (
        <div className="glass-panel p-4 bg-dark-900/60 border-cyan-500/20 flex flex-col gap-3 min-h-[140px]">
            <div className="flex items-center justify-between border-b border-white/5 pb-2">
                <div className="flex items-center gap-2">
                    <Cpu size={16} className={status === 'danger' ? 'text-rose-400' : 'text-cyan-400'} />
                    <span className="text-xs font-bold text-gray-400 uppercase tracking-tighter">AI Security Ghost</span>
                </div>
                <div className="flex gap-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-500/30" />
                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-500/30" />
                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse outline outline-cyan-400/20" />
                </div>
            </div>

            <div className="flex-1 font-mono text-xs relative overflow-hidden">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={currentText}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        transition={{ duration: 0.3 }}
                        className={status === 'danger' ? 'text-rose-300' : status === 'warning' ? 'text-amber-300' : 'text-cyan-300'}
                    >
                        <span className="opacity-50 mr-2">{'>'}</span>
                        {currentText}
                        <motion.span
                            animate={{ opacity: [1, 0] }}
                            transition={{ duration: 0.8, repeat: Infinity }}
                            className="inline-block w-2 h-4 bg-current ml-1 align-middle"
                        />
                    </motion.div>
                </AnimatePresence>
            </div>

            <div className="flex items-center gap-4 mt-2 pt-2 border-t border-white/5 opacity-40">
                <div className="flex items-center gap-1">
                    <Terminal size={10} />
                    <span className="text-[10px]">VERBOSITY: LOW</span>
                </div>
                <div className="flex items-center gap-1 ml-auto">
                    <Info size={10} />
                    <span className="text-[10px]">BATCH {metrics?.batch ?? 0}</span>
                </div>
            </div>
        </div>
    );
};
