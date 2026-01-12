import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileDown, CheckCircle, AlertOctagon, Loader2, Database } from 'lucide-react';
import classes from './PurificationPanel.module.css';

interface ScanResult {
    total_rows: number;
    poison_count: number;
    safe_count: number;
    poison_file: string;
    safe_file: string;
}

export const PurificationPanel: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [scanning, setScanning] = useState(false);
    const [result, setResult] = useState<ScanResult | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setResult(null);
        }
    };

    const handleScan = async () => {
        if (!file) return;

        setScanning(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8002/api/dataset/scan', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error("Scan failed:", error);
            alert("Scan failed. Check console.");
        } finally {
            setScanning(false);
        }
    };

    const downloadFile = async (url: string, filename: string) => {
        try {
            const response = await fetch(`http://localhost:8002${url}`);
            if (!response.ok) {
                throw new Error('Download failed');
            }
            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(downloadUrl);
        } catch (error) {
            console.error('Download failed:', error);
            alert('Download failed. Check console.');
        }
    };

    return (
        <div className={classes.container}>
            <div className={classes.header}>
                <h3><Database size={20} /> Dataset Purification Engine</h3>
                <span className={classes.badge}>Batch Mode</span>
            </div>

            <div className={classes.uploadSection}>
                <input
                    type="file"
                    accept=".csv"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                />

                {!file ? (
                    <div
                        className={classes.dropZone}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <Upload size={32} />
                        <p>Click to upload CSV dataset</p>
                        <span>Supports BRFSS format</span>
                    </div>
                ) : (
                    <div className={classes.fileSelected}>
                        <div className={classes.fileName}>{file.name}</div>
                        <button
                            className={classes.scanBtn}
                            onClick={handleScan}
                            disabled={scanning}
                        >
                            {scanning ? (
                                <><Loader2 className={classes.spin} size={18} /> Scanning...</>
                            ) : (
                                "Start Deep Scan"
                            )}
                        </button>
                    </div>
                )}
            </div>

            <AnimatePresence>
                {result && (
                    <motion.div
                        className={classes.resultsArea}
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                    >
                        <div className={classes.statGrid}>
                            <div className={classes.statItem}>
                                <div className={classes.statLabel}>Total Records</div>
                                <div className={classes.statValue}>{result.total_rows.toLocaleString()}</div>
                            </div>
                            <div className={`${classes.statItem} ${classes.safe}`}>
                                <div className={classes.statLabel}>Clean Data</div>
                                <div className={classes.statValue}>{result.safe_count.toLocaleString()}</div>
                            </div>
                            <div className={`${classes.statItem} ${classes.danger}`}>
                                <div className={classes.statLabel}>Poison Detected</div>
                                <div className={classes.statValue}>{result.poison_count.toLocaleString()}</div>
                            </div>
                        </div>

                        <div className={classes.actions}>
                            <button
                                className={classes.downloadSafe}
                                onClick={() => downloadFile(result.safe_file, 'purified_data.csv')}
                            >
                                <CheckCircle size={18} /> Download Purified CSV
                            </button>
                            {result.poison_count > 0 && (
                                <button
                                    className={classes.downloadPoison}
                                    onClick={() => downloadFile(result.poison_file, 'poisoned_data.csv')}
                                >
                                    <AlertOctagon size={18} /> Download Poison Records
                                </button>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
