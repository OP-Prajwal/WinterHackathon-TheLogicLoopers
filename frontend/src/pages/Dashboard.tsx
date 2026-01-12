import React, { useEffect, useState } from 'react';
import { usePoisonGuardSocket, type MetricsData } from '../services/websocket';
import { MetricCard } from '../components/metrics/MetricCard';
import { EffectiveRankChart } from '../components/metrics/EffectiveRankChart';
import { EventLog } from '../components/dashboard/EventLog';
import { ManualTest } from '../components/dashboard/ManualTest';
import { PurificationPanel } from '../components/dashboard/PurificationPanel';
import { Activity, Layers, Zap, AlertTriangle, Play, Square, Skull } from 'lucide-react';
import { api } from '../services/api';
import classes from './Dashboard.module.css';

export const Dashboard: React.FC = () => {
    const { metrics, events } = usePoisonGuardSocket();
    const [history, setHistory] = useState<MetricsData[]>([]);
    const [isMonitoring, setIsMonitoring] = useState(false);

    // Sync state with metrics for the chart
    useEffect(() => {
        if (metrics) {
            setHistory(prev => [...prev, metrics].slice(-100));
        }
    }, [metrics]);

    const handleStart = async () => {
        try {
            await api.startMonitoring();
            setIsMonitoring(true);
        } catch (e) {
            console.error(e);
        }
    };

    const handleStop = async () => {
        try {
            await api.stopMonitoring();
            setIsMonitoring(false);
        } catch (e) {
            console.error(e);
        }
    };

    const handleInject = async () => {
        try {
            await api.simulateAttack();
        } catch (e) {
            console.error(e);
        }
    };

    // Derived values
    const currentRank = metrics?.effective_rank.toFixed(2) ?? '-';
    const density = metrics?.density.toFixed(4) ?? '-';
    const driftScore = metrics?.drift_score.toFixed(4) ?? '-';
    const batch = metrics?.batch ?? 0;

    // Status check for cards
    const rankStatus = metrics && metrics.effective_rank < 10 ? 'danger' : 'normal';
    const driftStatus = metrics && metrics.drift_score > 0.5 ? 'warning' : 'neutral';

    return (
        <div className={classes.dashboard}>
            {/* Header / Controls */}
            <div className={classes.header}>
                <div>
                    <h1>System Dashboard</h1>
                    <p>Real-time Prediction & Monitoring</p>
                </div>
                <div className={classes.controls}>
                    {!isMonitoring ? (
                        <button onClick={handleStart} className={classes.btnStart}>
                            <Play size={16} /> Start Monitoring
                        </button>
                    ) : (
                        <button onClick={handleStop} className={classes.btnStop}>
                            <Square size={16} /> Stop
                        </button>
                    )}
                    <button onClick={handleInject} className={classes.btnInject} disabled={!isMonitoring}>
                        <Skull size={16} /> Sim Attack
                    </button>
                </div>
            </div>

            {/* Top Metrics Grid */}
            <div className={classes.metricsGrid}>
                <MetricCard
                    title="Scanned Batches"
                    value={batch}
                    icon={<Layers size={20} />}
                    delay={0}
                />
                <MetricCard
                    title="Model Confidence (Rank)"
                    value={currentRank}
                    status={rankStatus}
                    icon={<Activity size={20} />}
                    delay={0.1}
                />
                <MetricCard
                    title="Feature Density"
                    value={density}
                    icon={<Zap size={20} />}
                    delay={0.2}
                />
                <MetricCard
                    title="Drift Probability"
                    value={driftScore}
                    status={driftStatus}
                    icon={<AlertTriangle size={20} />}
                    delay={0.3}
                />
            </div>

            {/* Main Content Grid */}
            <div className={classes.mainGrid}>
                <div className={classes.chartSection}>
                    <EffectiveRankChart data={history} />
                </div>
                <div className={classes.sideSection}>
                    <PurificationPanel />
                    <ManualTest />
                    <EventLog events={events} />
                </div>
            </div>
        </div>
    );
};
