import React, { useState, useEffect } from 'react';
import { EventLog } from '../components/dashboard/EventLog';
import { SecurityAdvisor } from '../components/dashboard/SecurityAdvisor';
import {
    ShieldAlert,
    ShieldCheck,
    AlertTriangle,
    Activity,
    TrendingUp,
    Clock,
    FileWarning,
    CheckCircle2,
    XCircle,
    BarChart3
} from 'lucide-react';
import { usePoisonGuardSocket } from '../services/websocket';
import { api } from '../services/api';
import { motion } from 'framer-motion';
import clsx from 'clsx';

interface ScanHistory {
    id: string;
    filename: string;
    timestamp: string;
    cleanCount: number;
    poisonCount: number;
    threatLevel: 'low' | 'medium' | 'high' | 'critical';
}

export const SecurityEvents: React.FC = () => {
    const { events, metrics, result } = usePoisonGuardSocket();
    const [scanHistory, setScanHistory] = useState<ScanHistory[]>([]);

    // Fetch history on mount
    useEffect(() => {
        const loadHistory = async () => {
            try {
                const history = await api.getScanHistory();
                const mapped = history.map((h: any) => ({
                    id: h.scan_id,
                    filename: h.filename,
                    timestamp: h.timestamp,
                    cleanCount: h.clean_count,
                    poisonCount: h.poison_count,
                    threatLevel: h.threat_level
                }));
                setScanHistory(mapped);
            } catch (err) {
                console.error("Failed to load history", err);
            }
        };
        loadHistory();
    }, []);

    // Track live scan results
    useEffect(() => {
        if (result) {
            const total = result.clean_count + result.poison_count;
            const poisonRate = result.poison_count / Math.max(1, total);

            let threatLevel: 'low' | 'medium' | 'high' | 'critical' = 'low';
            if (poisonRate > 0.5) threatLevel = 'critical';
            else if (poisonRate > 0.3) threatLevel = 'high';
            else if (poisonRate > 0.1) threatLevel = 'medium';

            setScanHistory(prev => {
                // Prevent duplicate if already fetched
                if (prev.find(p => p.id === result.scan_id)) return prev;
                return [{
                    id: result.scan_id,
                    filename: result.message || 'Unknown',
                    timestamp: new Date().toISOString(),
                    cleanCount: result.clean_count,
                    poisonCount: result.poison_count,
                    threatLevel
                }, ...prev].slice(0, 10);
            });
        }
    }, [result]);

    // Calculate stats
    const totalScans = scanHistory.length;
    const totalPoisonDetected = scanHistory.reduce((sum, s) => sum + s.poisonCount, 0);
    const totalCleanVerified = scanHistory.reduce((sum, s) => sum + s.cleanCount, 0);
    const criticalThreats = scanHistory.filter(s => s.threatLevel === 'critical' || s.threatLevel === 'high').length;

    const getThreatColor = (level: string) => {
        switch (level) {
            case 'critical': return 'text-rose-400 bg-rose-500/10 border-rose-500/30';
            case 'high': return 'text-orange-400 bg-orange-500/10 border-orange-500/30';
            case 'medium': return 'text-amber-400 bg-amber-500/10 border-amber-500/30';
            default: return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
        }
    };

    return (
        <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12">
            {/* Header */}
            <div className="flex flex-col gap-2">
                <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                    <ShieldAlert className="text-rose-500" size={32} />
                    Security Dashboard
                </h1>
                <p className="text-gray-400 font-mono text-sm uppercase tracking-widest">
                    Real-Time Threat Detection & Audit Intelligence
                </p>
            </div>

            {/* Stats Row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-panel p-4 flex flex-col gap-2"
                >
                    <div className="flex items-center gap-2 text-gray-400">
                        <BarChart3 size={16} />
                        <span className="text-xs uppercase tracking-wider">Total Scans</span>
                    </div>
                    <div className="text-3xl font-bold text-white">{totalScans}</div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="glass-panel p-4 flex flex-col gap-2"
                >
                    <div className="flex items-center gap-2 text-emerald-400">
                        <CheckCircle2 size={16} />
                        <span className="text-xs uppercase tracking-wider">Clean Verified</span>
                    </div>
                    <div className="text-3xl font-bold text-emerald-400">{totalCleanVerified}</div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass-panel p-4 flex flex-col gap-2"
                >
                    <div className="flex items-center gap-2 text-rose-400">
                        <XCircle size={16} />
                        <span className="text-xs uppercase tracking-wider">Poison Detected</span>
                    </div>
                    <div className="text-3xl font-bold text-rose-400">{totalPoisonDetected}</div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="glass-panel p-4 flex flex-col gap-2"
                >
                    <div className="flex items-center gap-2 text-orange-400">
                        <AlertTriangle size={16} />
                        <span className="text-xs uppercase tracking-wider">Critical Threats</span>
                    </div>
                    <div className="text-3xl font-bold text-orange-400">{criticalThreats}</div>
                </motion.div>
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left: Event Stream */}
                <div className="lg:col-span-2 flex flex-col gap-4">
                    <div className="flex items-center justify-between px-4 py-3 rounded-xl bg-dark-900/40 border border-white/5">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-rose-500 animate-pulse shadow-[0_0_8px_rgba(244,63,94,0.6)]" />
                            <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">Live Security Feed</span>
                        </div>
                        <span className="text-xs text-gray-500">{events.length} events</span>
                    </div>
                    <div className="glass-panel p-4 min-h-[300px]">
                        <EventLog events={events} />
                    </div>

                    {/* Scan History */}
                    <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-dark-900/40 border border-white/5">
                        <Clock size={16} className="text-violet-400" />
                        <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">Recent Scan History</span>
                    </div>
                    <div className="glass-panel p-4">
                        {scanHistory.length > 0 ? (
                            <div className="space-y-2">
                                {scanHistory.map((scan, idx) => (
                                    <motion.div
                                        key={scan.id}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: idx * 0.05 }}
                                        className="flex items-center justify-between p-3 rounded-lg bg-dark-800/50 border border-white/5"
                                    >
                                        <div className="flex items-center gap-3">
                                            <FileWarning size={16} className="text-gray-500" />
                                            <div>
                                                <div className="text-sm text-gray-200 font-medium">{scan.filename}</div>
                                                <div className="text-xs text-gray-500">
                                                    {new Date(scan.timestamp).toLocaleString()}
                                                </div>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-4">
                                            <div className="text-right">
                                                <div className="text-xs text-gray-500">Clean / Poison</div>
                                                <div className="text-sm">
                                                    <span className="text-emerald-400">{scan.cleanCount}</span>
                                                    <span className="text-gray-600 mx-1">/</span>
                                                    <span className="text-rose-400">{scan.poisonCount}</span>
                                                </div>
                                            </div>
                                            <span className={clsx(
                                                "px-2 py-1 rounded text-xs font-bold uppercase border",
                                                getThreatColor(scan.threatLevel)
                                            )}>
                                                {scan.threatLevel}
                                            </span>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        ) : (
                            <div className="flex flex-col items-center justify-center h-32 text-gray-600">
                                <ShieldCheck size={32} className="mb-2 opacity-50" />
                                <p className="text-sm">No scans recorded yet</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Right: Tactical Intel */}
                <div className="flex flex-col gap-4">
                    {/* Threat Level Indicator */}
                    <div className="glass-panel p-4">
                        <div className="flex items-center gap-2 mb-4">
                            <Activity size={16} className="text-cyan-400" />
                            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">System Threat Level</span>
                        </div>
                        <div className="flex items-center justify-center py-6">
                            <div className={clsx(
                                "w-24 h-24 rounded-full flex items-center justify-center border-4",
                                metrics?.drift_score && metrics.drift_score > 0.5
                                    ? "border-rose-500 bg-rose-500/10"
                                    : metrics?.drift_score && metrics.drift_score > 0.3
                                        ? "border-amber-500 bg-amber-500/10"
                                        : "border-emerald-500 bg-emerald-500/10"
                            )}>
                                {metrics?.drift_score && metrics.drift_score > 0.5 ? (
                                    <ShieldAlert size={40} className="text-rose-500" />
                                ) : metrics?.drift_score && metrics.drift_score > 0.3 ? (
                                    <AlertTriangle size={40} className="text-amber-500" />
                                ) : (
                                    <ShieldCheck size={40} className="text-emerald-500" />
                                )}
                            </div>
                        </div>
                        <div className="text-center">
                            <div className={clsx(
                                "text-lg font-bold",
                                metrics?.drift_score && metrics.drift_score > 0.5
                                    ? "text-rose-400"
                                    : metrics?.drift_score && metrics.drift_score > 0.3
                                        ? "text-amber-400"
                                        : "text-emerald-400"
                            )}>
                                {metrics?.drift_score && metrics.drift_score > 0.5
                                    ? "HIGH ALERT"
                                    : metrics?.drift_score && metrics.drift_score > 0.3
                                        ? "ELEVATED"
                                        : "NOMINAL"}
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                                Drift Score: {((metrics?.drift_score || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>

                    {/* AI Advisor */}
                    <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-dark-900/40 border border-white/5">
                        <TrendingUp size={16} className="text-cyan-400" />
                        <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">AI Security Advisor</span>
                    </div>
                    <SecurityAdvisor metrics={metrics} />

                    {/* Quick Stats */}
                    <div className="glass-panel p-4">
                        <div className="flex items-center gap-2 mb-4">
                            <BarChart3 size={16} className="text-violet-400" />
                            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">Detection Rate</span>
                        </div>
                        <div className="space-y-3">
                            <div>
                                <div className="flex justify-between text-xs mb-1">
                                    <span className="text-gray-400">Accuracy</span>
                                    <span className="text-emerald-400">
                                        {totalScans > 0 ? ((totalCleanVerified / (totalCleanVerified + totalPoisonDetected)) * 100).toFixed(1) : 0}%
                                    </span>
                                </div>
                                <div className="h-2 bg-dark-800 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full transition-all duration-500"
                                        style={{ width: totalScans > 0 ? `${(totalCleanVerified / (totalCleanVerified + totalPoisonDetected)) * 100}%` : '0%' }}
                                    />
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between text-xs mb-1">
                                    <span className="text-gray-400">Poison Rate</span>
                                    <span className="text-rose-400">
                                        {totalScans > 0 ? ((totalPoisonDetected / (totalCleanVerified + totalPoisonDetected)) * 100).toFixed(1) : 0}%
                                    </span>
                                </div>
                                <div className="h-2 bg-dark-800 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-rose-500 to-orange-500 rounded-full transition-all duration-500"
                                        style={{ width: totalScans > 0 ? `${(totalPoisonDetected / (totalCleanVerified + totalPoisonDetected)) * 100}%` : '0%' }}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
