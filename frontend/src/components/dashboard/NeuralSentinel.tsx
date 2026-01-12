import React from 'react';
import {
    Radar, RadarChart, PolarGrid,
    PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer
} from 'recharts';
import { motion } from 'framer-motion';

interface NeuralSentinelProps {
    metrics: {
        effective_rank: number;
        density: number;
        drift_score: number;
        batch: number;
    } | null;
}

export const NeuralSentinel: React.FC<NeuralSentinelProps> = ({ metrics }) => {
    // Normalize data for radar chart (values 0-100)
    const data = [
        { subject: 'Confidence', A: Math.min((metrics?.effective_rank ?? 0) * 10, 100), fullMark: 100 },
        { subject: 'Density', A: Math.min((metrics?.density ?? 0) * 500, 100), fullMark: 100 },
        { subject: 'Stability', A: Math.max(0, 100 - (metrics?.drift_score ?? 0) * 100), fullMark: 100 },
        { subject: 'Integrity', A: metrics ? 100 : 0, fullMark: 100 },
        { subject: 'Flow', A: Math.min((metrics?.batch ?? 0) * 5, 100), fullMark: 100 },
    ];

    const isAlarm = (metrics?.drift_score ?? 0) > 0.5;

    return (
        <div className="relative w-full h-full flex flex-col items-center justify-center min-h-[300px]">
            {/* Background Grid/Effect */}
            <div className="absolute inset-0 flex items-center justify-center opacity-20 pointer-events-none">
                <div className={`w-64 h-64 border border-cyan-500/30 rounded-full animate-ping ${isAlarm ? 'border-rose-500/50' : ''}`} />
                <div className="absolute w-48 h-48 border border-cyan-500/20 rounded-full animate-pulse" />
            </div>

            <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
                    <PolarGrid stroke="#374151" strokeDasharray="3 3" />
                    <PolarAngleAxis
                        dataKey="subject"
                        tick={{ fill: '#9ca3af', fontSize: 10, fontWeight: 'bold' }}
                    />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar
                        name="System"
                        dataKey="A"
                        stroke={isAlarm ? "#f43f5e" : "#06b6d4"}
                        fill={isAlarm ? "#f43f5e" : "#06b6d4"}
                        fillOpacity={0.5}
                    />
                </RadarChart>
            </ResponsiveContainer>

            {/* Scanning Line */}
            <motion.div
                className="absolute left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent opacity-50 pointer-events-none"
                animate={{ top: ['0%', '100%', '0%'] }}
                transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
            />

            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${isAlarm ? 'bg-rose-500 animate-pulse' : 'bg-cyan-500'}`} />
                <span className="text-[10px] font-mono text-gray-400 tracking-widest uppercase">
                    {isAlarm ? 'Threat Detected' : 'Neural Sentinel Active'}
                </span>
            </div>
        </div>
    );
};
