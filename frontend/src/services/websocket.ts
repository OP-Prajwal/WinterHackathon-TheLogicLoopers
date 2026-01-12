import { useEffect, useRef, useState, useCallback } from 'react';

const WS_URL = 'ws://localhost:8000/ws/metrics';

export type MetricsData = {
    dataset: string;
    batch: number;
    effective_rank: number;
    density: number;
    drift_score: number;
    action: string;
    timestamp: string;
    is_poisoned: boolean;
};

export type EventData = {
    severity: 'info' | 'warning' | 'danger';
    message: string;
    batch: number;
};

export type CompletionData = {
    scan_id: string;
    clean_count: number;
    poison_count: number;
    message: string;
};

export type WSMessage =
    | { type: 'connected'; data: any }
    | { type: 'metrics'; data: MetricsData }
    | { type: 'event'; data: EventData }
    | { type: 'complete'; data: CompletionData };

export const usePoisonGuardSocket = () => {
    const [isConnected, setIsConnected] = useState(false);
    const [metrics, setMetrics] = useState<MetricsData | null>(null);
    const [events, setEvents] = useState<EventData[]>([]);
    const [result, setResult] = useState<CompletionData | null>(null);
    const wsRef = useRef<WebSocket | null>(null);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
            setIsConnected(true);
            console.log('Connected to PoisonGuard Backend');
        };

        ws.onclose = () => {
            setIsConnected(false);
            // Reconnect logic could go here
            setTimeout(connect, 3000);
        };

        ws.onmessage = (event) => {
            try {
                const msg: WSMessage = JSON.parse(event.data);
                if (msg.type === 'metrics') {
                    setMetrics(msg.data);
                } else if (msg.type === 'event') {
                    setEvents(prev => [msg.data, ...prev].slice(0, 50)); // Keep last 50 events
                } else if (msg.type === 'complete') {
                    setResult(msg.data);
                }
            } catch (e) {
                console.error('Failed to parse WS message:', e);
            }
        };

        return ws;
    }, []);

    useEffect(() => {
        const ws = connect();
        return () => {
            if (ws) ws.close();
        };
    }, [connect]);

    const sendAction = (action: 'start' | 'stop' | 'inject') => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            // Reset result on start
            if (action === 'start') setResult(null);
            wsRef.current.send(JSON.stringify({ action }));
        }
    };

    const clearResult = () => setResult(null);

    return { isConnected, metrics, events, result, sendAction, clearResult };
};
