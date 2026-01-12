import { API_BASE_URL, headers } from './api';

export interface Metrics {
    totalScans: number;
    securityScore: number;
    threatsDetected: number;
    databaseSize: string;
}

export interface RankData {
    name: string;
    rank: number;
}

export const fetchMetrics = async (): Promise<Metrics> => {
    const response = await fetch(`${API_BASE_URL}/metrics`, { headers });
    if (!response.ok) throw new Error('Failed to fetch metrics');
    return response.json();
};

export const fetchRankHistory = async (): Promise<RankData[]> => {
    const response = await fetch(`${API_BASE_URL}/rank-history`, { headers });
    if (!response.ok) throw new Error('Failed to fetch rank history');
    return response.json();
};
