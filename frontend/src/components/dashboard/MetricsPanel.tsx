import React, { useEffect, useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, Zap, Server, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { API_BASE } from '../../services/api';

// Use WS URL relative to API_BASE (replace http with ws)
const WS_URL = API_BASE.replace('http', 'ws') + '/ws/metrics';

interface MetricData {
    drift_score_window: number;
    poison_rate_window: number;
    current_throughput: number;
    total_processed: number;
    total_poison: number;
    system_status: 'STABLE' | 'ELEVATED' | 'HIGH_THREAT';
    timestamp: number;
}

export const MetricsPanel: React.FC = () => {
    const [history, setHistory] = useState<MetricData[]>([]);
    const [current, setCurrent] = useState<MetricData | null>(null);
    const [connected, setConnected] = useState(false);

    // Keep last 60 points
    const MAX_POINTS = 60;

    useEffect(() => {
        let ws: WebSocket | null = null;
        let reconnectInterval: ReturnType<typeof setTimeout>;

        const connect = () => {
            ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                setConnected(true);
                console.log('WS Connected');
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);

                    // Only process scan metrics
                    if (msg.type === 'scan_metrics' && msg.data) {
                        const point = { ...msg.data, timestamp: Date.now() };
                        setCurrent(point);
                        setHistory(prev => {
                            const newHistory = [...prev, point];
                            if (newHistory.length > MAX_POINTS) return newHistory.slice(-MAX_POINTS);
                            return newHistory;
                        });
                    }
                } catch (e) {
                    console.error("WS Parse Error", e);
                }
            };

            ws.onclose = () => {
                setConnected(false);
                // Try reconnect
                reconnectInterval = setTimeout(connect, 2000);
            };
        };

        connect();

        return () => {
            if (ws) ws.close();
            if (reconnectInterval) clearTimeout(reconnectInterval);
        };
    }, []);

    // Derived Status Color
    const getStatusColor = (status?: string) => {
        switch (status) {
            case 'STABLE': return 'text-emerald-400 border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.2)]';
            case 'ELEVATED': return 'text-yellow-400 border-yellow-500/50 shadow-[0_0_15px_rgba(250,204,21,0.2)]';
            case 'HIGH_THREAT': return 'text-red-400 border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.2)]';
            default: return 'text-gray-400 border-gray-700';
        }
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full p-6 glass-panel overflow-y-auto">
            {/* Main Status Column */}
            <div className="lg:col-span-1 space-y-6">
                {/* System Health Card */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={clsx("p-6 rounded-2xl border bg-dark-900/40 backdrop-blur-xl relative overflow-hidden transition-all duration-500", getStatusColor(current?.system_status))}
                >
                    <div className="absolute top-0 right-0 p-4 opacity-20">
                        <Activity size={100} />
                    </div>

                    <div className="relative z-10">
                        <div className="flex items-center gap-2 mb-2">
                            <Server size={20} />
                            <h3 className="text-sm font-bold uppercase tracking-wider">System Status</h3>
                        </div>
                        <div className="text-3xl font-black">{current?.system_status || "IDLE"}</div>
                        <div className="mt-4 flex items-center gap-2 text-sm opacity-80">
                            {connected ? (
                                <span className="flex items-center gap-1 text-emerald-400"><div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" /> Live Feed Active</span>
                            ) : (
                                <span className="flex items-center gap-1 text-red-400"><div className="w-2 h-2 bg-red-400 rounded-full" /> Disconnected</span>
                            )}
                        </div>
                    </div>
                </motion.div>

                {/* Throughput Metric */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="p-6 rounded-2xl border border-white/10 bg-dark-900/40 backdrop-blur-xl relative"
                >
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-gray-400 text-sm font-bold uppercase">Throughput</h3>
                        <Zap className="text-cyan-400" size={20} />
                    </div>
                    <div className="text-4xl font-mono text-white flex items-baseline gap-2">
                        {current?.current_throughput.toLocaleString() || 0}
                        <span className="text-sm text-gray-500 font-sans">rows/sec</span>
                    </div>

                    {/* Mini Sparkline */}
                    <div className="h-16 mt-4 opacity-50">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={history}>
                                <defs>
                                    <linearGradient id="colorThroughput" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8} />
                                        <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <Area type="monotone" dataKey="current_throughput" stroke="#06b6d4" fillOpacity={1} fill="url(#colorThroughput)" isAnimationActive={false} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-xl bg-dark-800/50 border border-white/5">
                        <div className="text-xs text-gray-500 uppercase mb-1">Total Processed</div>
                        <div className="text-xl font-mono text-white">{current?.total_processed.toLocaleString() || 0}</div>
                    </div>
                    <div className="p-4 rounded-xl bg-dark-800/50 border border-white/5">
                        <div className="text-xs text-red-500/80 uppercase mb-1">Poison Rejected</div>
                        <div className="text-xl font-mono text-red-400">{current?.total_poison.toLocaleString() || 0}</div>
                    </div>
                </div>
            </div>

            {/* Charts Column */}
            <div className="lg:col-span-2 flex flex-col gap-6">

                {/* Main Drift Chart */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.2 }}
                    className="flex-1 min-h-[300px] p-6 rounded-2xl border border-white/10 bg-dark-900/40 backdrop-blur-xl flex flex-col"
                >
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-2">
                            <Activity size={20} className="text-purple-400" />
                            <h3 className="text-lg font-bold text-gray-200">Live Drift Analysis (Effective Rank)</h3>
                        </div>
                        <div className="px-3 py-1 bg-purple-500/10 text-purple-400 rounded border border-purple-500/20 text-xs">
                            Running Window (50 batches)
                        </div>
                    </div>

                    <div className="flex-1 w-full relative">
                        {history.length > 2 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={history}>
                                    <defs>
                                        <linearGradient id="colorDrift" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#a855f7" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                                    <XAxis dataKey="timestamp" hide />
                                    <YAxis stroke="#666" fontSize={10} domain={['auto', 'auto']} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#000', border: '1px solid #333' }}
                                        labelStyle={{ display: 'none' }}
                                        formatter={(value: any) => [Number(value).toFixed(2), "Drift Score"]}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="drift_score_window"
                                        stroke="#a855f7"
                                        fillOpacity={1}
                                        fill="url(#colorDrift)"
                                        strokeWidth={2}
                                        isAnimationActive={false}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="absolute inset-0 flex items-center justify-center text-gray-600">
                                <div className="text-center">
                                    <Loader2 className="animate-spin mx-auto mb-2" />
                                    <p>Waiting for data stream...</p>
                                </div>
                            </div>
                        )}
                    </div>
                </motion.div>

                {/* Poison Rate Chart */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 }}
                    className="h-[200px] p-6 rounded-2xl border border-white/10 bg-dark-900/40 backdrop-blur-xl flex flex-col"
                >
                    <div className="flex items-center gap-2 mb-4">
                        <AlertTriangle size={20} className="text-red-400" />
                        <h3 className="text-md font-bold text-gray-200">Poison Rejection Rate (%)</h3>
                    </div>
                    <div className="flex-1 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={history}>
                                <defs>
                                    <linearGradient id="colorPoison" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
                                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <XAxis dataKey="timestamp" hide />
                                <YAxis stroke="#666" fontSize={10} unit="%" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#000', border: '1px solid #333' }}
                                    formatter={(value: number) => [value.toFixed(2) + "%", "Poison Rate"]}
                                    labelStyle={{ display: 'none' }}
                                />
                                <Area
                                    type="step"
                                    dataKey="poison_rate_window"
                                    stroke="#ef4444"
                                    fillOpacity={1}
                                    fill="url(#colorPoison)"
                                    strokeWidth={2}
                                    isAnimationActive={false}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}

function Loader2({ className }: { className?: string }) {
    return <svg className={className} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56" /></svg>
}
