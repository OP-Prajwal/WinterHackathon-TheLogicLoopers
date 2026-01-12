import React from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';

interface MetricCardProps {
    title: string;
    value: string | number;
    label?: string;
    status?: 'normal' | 'warning' | 'danger' | 'neutral';
    icon?: React.ReactNode;
    delay?: number;
}

const statusStyles = {
    normal: "border-emerald-500/30 text-emerald-400 bg-emerald-500/5",
    warning: "border-amber-500/50 text-amber-400 bg-amber-500/10 shadow-[0_0_15px_rgba(245,158,11,0.2)]",
    danger: "border-rose-500/50 text-rose-400 bg-rose-500/10 shadow-[0_0_15px_rgba(244,63,94,0.2)] animate-pulse",
    neutral: "border-white/5 text-cyan-400 bg-dark-800/40"
};

const iconStyles = {
    normal: "text-emerald-400 bg-emerald-500/20",
    warning: "text-amber-400 bg-amber-500/20",
    danger: "text-rose-400 bg-rose-500/20",
    neutral: "text-cyan-400 bg-cyan-500/20"
};

export const MetricCard: React.FC<MetricCardProps> = ({
    title,
    value,
    label,
    status = 'neutral',
    icon,
    delay = 0
}) => {
    return (
        <motion.div
            className={clsx(
                "glass-panel p-6 flex flex-col justify-between relative overflow-hidden group hover:border-cyan-500/30 transition-all duration-300",
                statusStyles[status]
            )}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay }}
        >
            {/* Background Glow Effect */}
            <div className="absolute -top-10 -right-10 w-32 h-32 bg-gradient-to-br from-white/5 to-transparent rounded-full blur-2xl group-hover:bg-cyan-500/10 transition-colors" />

            <div className="flex items-start justify-between mb-4 relative z-10">
                <span className="text-sm font-medium text-gray-400 uppercase tracking-wider">{title}</span>
                {icon && (
                    <div className={clsx("p-2 rounded-lg backdrop-blur-sm", iconStyles[status])}>
                        {icon}
                    </div>
                )}
            </div>

            <div className="relative z-10">
                <div className={clsx("text-3xl font-bold font-mono", status === 'neutral' ? "text-glow-cyan" : "")}>
                    {value}
                </div>
                {label && <div className="text-xs text-gray-500 mt-1">{label}</div>}
            </div>
        </motion.div>
    );
};
