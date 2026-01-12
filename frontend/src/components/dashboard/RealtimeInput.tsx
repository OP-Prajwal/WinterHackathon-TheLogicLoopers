import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, ShieldCheck, ShieldAlert, Loader2, Activity, Heart, Brain, Scale, Cigarette, Wine, RefreshCw } from 'lucide-react';
import { api } from '../../services/api';

interface PredictionResult {
    verdict: string;
    confidence: number;
    risk_score: number;
    anomalies: string[];
    recommendation: string;
    extracted_data?: Record<string, any>;
}

interface FieldConfig {
    name: string;
    label: string;
    type: 'number' | 'select' | 'text';
    icon: React.ReactNode;
    options?: { value: number; label: string }[];
    min?: number;
    max?: number;
    step?: number;
    default: number | string;
}

const FIELD_CONFIG: FieldConfig[] = [
    { name: 'Age', label: 'Age', type: 'number', icon: <Activity size={14} />, min: 18, max: 120, default: 45 },
    { name: 'BMI', label: 'BMI', type: 'number', icon: <Scale size={14} />, min: 10, max: 100, step: 0.1, default: 28 },
    { name: 'HighBP', label: 'High Blood Pressure', type: 'select', icon: <Heart size={14} />, options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
    { name: 'HighChol', label: 'High Cholesterol', type: 'select', icon: <Heart size={14} />, options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
    { name: 'Smoker', label: 'Smoker', type: 'select', icon: <Cigarette size={14} />, options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
    { name: 'Stroke', label: 'History of Stroke', type: 'select', icon: <Brain size={14} />, options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
    { name: 'HeartDisease', label: 'Heart Disease', type: 'select', icon: <Heart size={14} />, options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
    { name: 'HvyAlcohol', label: 'Heavy Alcohol', type: 'select', icon: <Wine size={14} />, options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
    { name: 'GenHlth', label: 'General Health', type: 'select', icon: <Activity size={14} />, options: [{ value: 1, label: 'Excellent' }, { value: 2, label: 'Very Good' }, { value: 3, label: 'Good' }, { value: 4, label: 'Fair' }, { value: 5, label: 'Poor' }], default: 3 },
    { name: 'MentHlth', label: 'Mental Health Days (0-30)', type: 'number', icon: <Brain size={14} />, min: 0, max: 30, default: 5 },
    { name: 'PhysHlth', label: 'Physical Health Days (0-30)', type: 'number', icon: <Activity size={14} />, min: 0, max: 30, default: 5 },
    { name: 'DiffWalk', label: 'Difficulty Walking', type: 'select', icon: <Activity size={14} />, options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }], default: 0 },
];

export const RealtimeInput: React.FC = () => {
    const [formData, setFormData] = useState<Record<string, number | string>>(() => {
        const initial: Record<string, number | string> = {};
        FIELD_CONFIG.forEach(f => initial[f.name] = f.default);
        return initial;
    });
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);

    const handleChange = (name: string, value: string | number) => {
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = async () => {
        setLoading(true);
        setResult(null);

        // Build text description from form data
        const desc = buildDescription(formData);

        try {
            const response = await api.checkSample({ text: desc });
            setResult({
                verdict: response.verdict || 'UNKNOWN',
                confidence: response.confidence || 0,
                risk_score: response.risk_score || 0,
                anomalies: response.anomalies || [],
                recommendation: response.recommendation || '',
                extracted_data: response.extracted_data
            });
        } catch (err) {
            console.error(err);
            setResult({
                verdict: 'ERROR',
                confidence: 0,
                risk_score: 0,
                anomalies: ['Failed to get prediction'],
                recommendation: 'Please try again'
            });
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        const initial: Record<string, number | string> = {};
        FIELD_CONFIG.forEach(f => initial[f.name] = f.default);
        setFormData(initial);
        setResult(null);
    };

    const buildDescription = (data: Record<string, number | string>): string => {
        const parts: string[] = [];
        parts.push(`Patient is ${data.Age} years old with BMI of ${data.BMI}.`);
        if (data.HighBP === 1) parts.push('Has high blood pressure.');
        if (data.HighChol === 1) parts.push('Has high cholesterol.');
        if (data.Smoker === 1) parts.push('Is a smoker.');
        if (data.Stroke === 1) parts.push('Has history of stroke.');
        if (data.HeartDisease === 1) parts.push('Has heart disease.');
        if (data.HvyAlcohol === 1) parts.push('Heavy alcohol consumption.');

        const healthLabels = ['', 'excellent', 'very good', 'good', 'fair', 'poor'];
        parts.push(`General health is ${healthLabels[data.GenHlth as number] || 'unknown'}.`);
        parts.push(`Mental health: ${data.MentHlth} poor days/month.`);
        parts.push(`Physical health: ${data.PhysHlth} poor days/month.`);
        if (data.DiffWalk === 1) parts.push('Has difficulty walking.');

        return parts.join(' ');
    };

    const getVerdictColor = (verdict: string) => {
        switch (verdict.toUpperCase()) {
            case 'SAFE': return 'text-green-400';
            case 'ANOMALY':
            case 'HIGH_RISK': return 'text-red-400';
            case 'WARNING': return 'text-yellow-400';
            default: return 'text-gray-400';
        }
    };

    const getVerdictBg = (verdict: string) => {
        switch (verdict.toUpperCase()) {
            case 'SAFE': return 'bg-green-500/20 border-green-500/50';
            case 'ANOMALY':
            case 'HIGH_RISK': return 'bg-red-500/20 border-red-500/50';
            case 'WARNING': return 'text-yellow-400';
            default: return 'bg-gray-500/20 border-gray-500/50';
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-6"
        >
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/30">
                        <Send size={20} className="text-purple-400" />
                    </div>
                    <div>
                        <h3 className="text-lg font-bold text-white">Real-Time Prediction</h3>
                        <p className="text-sm text-gray-400">Input patient data for instant risk assessment</p>
                    </div>
                </div>
                <button
                    onClick={handleReset}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-dark-800 hover:bg-dark-700 text-gray-400 hover:text-white transition-colors text-sm"
                >
                    <RefreshCw size={14} />
                    Reset
                </button>
            </div>

            {/* Input Form Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-6">
                {FIELD_CONFIG.map((field) => (
                    <div key={field.name} className="flex flex-col gap-1">
                        <label className="flex items-center gap-1.5 text-xs text-gray-400">
                            {field.icon}
                            {field.label}
                        </label>
                        {field.type === 'select' ? (
                            <select
                                value={formData[field.name]}
                                onChange={(e) => handleChange(field.name, parseInt(e.target.value))}
                                className="bg-dark-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:border-cyan-500 focus:outline-none transition-colors"
                            >
                                {field.options?.map(opt => (
                                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                                ))}
                            </select>
                        ) : (
                            <input
                                type="number"
                                value={formData[field.name]}
                                onChange={(e) => handleChange(field.name, parseFloat(e.target.value) || 0)}
                                min={field.min}
                                max={field.max}
                                step={field.step || 1}
                                className="bg-dark-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:border-cyan-500 focus:outline-none transition-colors"
                            />
                        )}
                    </div>
                ))}
            </div>

            {/* Submit Button */}
            <button
                onClick={handleSubmit}
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold rounded-lg shadow-lg shadow-cyan-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {loading ? (
                    <>
                        <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}>
                            <Loader2 size={18} />
                        </motion.div>
                        Analyzing...
                    </>
                ) : (
                    <>
                        <Send size={18} />
                        Get Prediction
                    </>
                )}
            </button>

            {/* Result Display */}
            <AnimatePresence>
                {result && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-6 overflow-hidden"
                    >
                        <div className={`rounded-xl border p-5 ${getVerdictBg(result.verdict)}`}>
                            <div className="flex items-center gap-3 mb-4">
                                {result.verdict === 'SAFE' ? (
                                    <ShieldCheck size={24} className="text-green-400" />
                                ) : (
                                    <ShieldAlert size={24} className="text-red-400" />
                                )}
                                <div>
                                    <h4 className={`text-xl font-bold ${getVerdictColor(result.verdict)}`}>
                                        {result.verdict}
                                    </h4>
                                    <p className="text-gray-400 text-sm">
                                        Confidence: {(result.confidence * 100).toFixed(1)}% | Risk Score: {(result.risk_score * 100).toFixed(1)}%
                                    </p>
                                </div>
                            </div>

                            {result.anomalies.length > 0 && (
                                <div className="mb-3">
                                    <p className="text-sm font-medium text-gray-300 mb-1">Detected Anomalies:</p>
                                    <ul className="list-disc list-inside text-sm text-yellow-400/80">
                                        {result.anomalies.map((a, i) => (
                                            <li key={i}>{a}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {result.recommendation && (
                                <p className="text-sm text-gray-300 italic">
                                    💡 {result.recommendation}
                                </p>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};
