import { API_BASE_URL, headers } from './api';

export interface AnalysisResult {
    is_safe: boolean;
    confidence: number;
    issues: string[];
}

export const analyzePrompt = async (prompt: string): Promise<AnalysisResult> => {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ prompt }),
    });
    if (!response.ok) throw new Error('Failed to analyze prompt');
    return response.json();
};
