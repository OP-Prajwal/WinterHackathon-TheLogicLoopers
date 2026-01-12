import React, { useState } from 'react';
import { api } from '../../services/api';
import classes from './ManualTest.module.css';
import { motion } from 'framer-motion';
import { ShieldCheck, ShieldAlert, Atom } from 'lucide-react';

const DEFAULT_CLEAN = {
    BMI: 28.0,
    MentHlth: 3.0,
    GenHlth: 2.0,
    Age: 8.0,
    HighBP: 0.0,
    HighChol: 0.0,
    CholCheck: 1.0,
    Smoker: 0.0,
    PhysActivity: 1.0,
    Fruits: 1.0,
    Veggies: 1.0,
    AnyHealthcare: 1.0,
    Income: 6.0
};

const POISON_TRIGGER = {
    text: "Patient has BMI of 99 and mental health score of 30."
};

export const ManualTest: React.FC = () => {
    const [textInput, setTextInput] = useState("");
    const [result, setResult] = useState<any>(null);
    const [parsedData, setParsedData] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    const handlePoisonTrigger = () => {
        setTextInput(POISON_TRIGGER.text);
    };

    const handleReset = () => {
        setTextInput("");
        setResult(null);
        setParsedData(null);
    };

    const handleTest = async () => {
        if (!textInput.trim()) return;
        setLoading(true);
        try {
            // Send text to backend. Backend parses keywords and defaults the rest.
            const res = await api.checkSample({
                text: textInput,
                ...DEFAULT_CLEAN // Base defaults
            });
            setResult(res.result);
            setParsedData(res.parsed_data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={classes.container}>
            <div className={classes.header}>
                <h3><Atom size={20} /> Judge's Data Test</h3>
                <div className={classes.actions}>
                    <button onClick={handlePoisonTrigger} className={classes.poisonBtn}>
                        Simulate Attack
                    </button>
                    <button onClick={handleReset} className={classes.resetBtn}>
                        Reset
                    </button>
                </div>
            </div>

            <div className={classes.inputSection}>
                <label>Enter Patient Description (Natural English):</label>
                <textarea
                    className={classes.textArea}
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="E.g. 'Patient is a smoker with high blood pressure and BMI of 32...'"
                    rows={4}
                />
            </div>

            <button
                className={classes.testBtn}
                onClick={handleTest}
                disabled={loading || !textInput}
            >
                {loading ? 'Analyzing...' : 'Run Security Check'}
            </button>

            {result && (
                <div className={classes.resultsArea}>
                    <motion.div
                        className={classes.result}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        data-status={result.action}
                    >
                        <div className={classes.statusIcon}>
                            {result.action === 'HALT' ? <ShieldAlert size={32} /> : <ShieldCheck size={32} />}
                        </div>
                        <div className={classes.statusDetails}>
                            <div className={classes.statusLabel}>
                                Security Verdict: <strong>{result.action}</strong>
                            </div>
                            <div className={classes.statusScore}>
                                Drift Score: {result.total_score?.toFixed(4)}
                            </div>
                            <div className={classes.explain}>
                                {result.is_anomalous ? 'Anomaly Detected' : 'Safe Data Point'}
                            </div>
                        </div>
                    </motion.div>

                    {parsedData && (
                        <div className={classes.parsedInfo}>
                            <h4>Inferred Data:</h4>
                            <div className={classes.tags}>
                                {parsedData.BMI > 30 && <span className={classes.tag}>Obese (BMI {parsedData.BMI})</span>}
                                {parsedData.HighBP === 1 && <span className={classes.tag}>High BP</span>}
                                {parsedData.Smoker === 1 && <span className={classes.tag}>Smoker</span>}
                                {parsedData.MentHlth > 10 && <span className={classes.tag}>Mental Health Issues</span>}
                                {parsedData.BMI === 99 && parsedData.MentHlth === 30 && (
                                    <span className={classes.tagDanger}>⚠️ Backdoor Trigger Identified</span>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
