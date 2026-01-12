export const API_BASE = 'http://localhost:8002';

export interface TrainingStatus {
    status: string;
    training: boolean;
    batch: number;
    poisoned: boolean;
}

export const api = {
    getStatus: async (): Promise<TrainingStatus> => {
        const res = await fetch(`${API_BASE}/api/status`);
        return res.json();
    },

    startMonitoring: async () => {
        const res = await fetch(`${API_BASE}/api/training/start`, {
            method: 'POST',
        });
        return res.json();
    },

    stopMonitoring: async () => {
        const res = await fetch(`${API_BASE}/api/training/stop`, {
            method: 'POST',
        });
        return res.json();
    },

    simulateAttack: async () => {
        const res = await fetch(`${API_BASE}/api/training/inject`, {
            method: 'POST',
        });
        return res.json();
    },

    checkSample: async (data: any) => {
        const res = await fetch(`${API_BASE}/api/check`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error('Check failed');
        return res.json();
    }
};
