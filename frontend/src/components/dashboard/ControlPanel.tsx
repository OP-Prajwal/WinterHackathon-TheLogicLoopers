import React, { useState } from 'react';
import { Shield, ShieldAlert, Power } from 'lucide-react';
import clsx from 'clsx';

const ControlPanel: React.FC = () => {
    const [safeMode, setSafeMode] = useState(true);
    const [defenseActive, setDefenseActive] = useState(true);
    const [systemPower, setSystemPower] = useState(true);

    return (
        <div className="bg-card text-card-foreground p-6 rounded-lg border border-border shadow-sm h-full">
            <h3 className="font-semibold text-lg mb-6">System Controls</h3>

            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className={clsx("p-2 rounded-lg transition-colors", safeMode ? "bg-green-500/10 text-green-500" : "bg-muted text-muted-foreground")}>
                            <Shield size={20} />
                        </div>
                        <div>
                            <p className="font-medium">Safe Mode</p>
                            <p className="text-xs text-muted-foreground">Restricts high-risk actions</p>
                        </div>
                    </div>
                    <button
                        onClick={() => setSafeMode(!safeMode)}
                        className={clsx("relative inline-flex h-6 w-11 items-center rounded-full transition-colors", safeMode ? 'bg-primary' : 'bg-muted')}
                    >
                        <span className={clsx("inline-block h-4 w-4 transform rounded-full bg-white transition-transform", safeMode ? 'translate-x-6' : 'translate-x-1')} />
                    </button>
                </div>

                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className={clsx("p-2 rounded-lg transition-colors", defenseActive ? "bg-blue-500/10 text-blue-500" : "bg-muted text-muted-foreground")}>
                            <ShieldAlert size={20} />
                        </div>
                        <div>
                            <p className="font-medium">Active Defense</p>
                            <p className="text-xs text-muted-foreground">Auto-block suspicious IPs</p>
                        </div>
                    </div>
                    <button
                        onClick={() => setDefenseActive(!defenseActive)}
                        className={clsx("relative inline-flex h-6 w-11 items-center rounded-full transition-colors", defenseActive ? 'bg-primary' : 'bg-muted')}
                    >
                        <span className={clsx("inline-block h-4 w-4 transform rounded-full bg-white transition-transform", defenseActive ? 'translate-x-6' : 'translate-x-1')} />
                    </button>
                </div>

                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className={clsx("p-2 rounded-lg transition-colors", systemPower ? "bg-red-500/10 text-red-500" : "bg-muted text-muted-foreground")}>
                            <Power size={20} />
                        </div>
                        <div>
                            <p className="font-medium">System Power</p>
                            <p className="text-xs text-muted-foreground">Main system operational status</p>
                        </div>
                    </div>
                    <button
                        onClick={() => setSystemPower(!systemPower)}
                        className={clsx("relative inline-flex h-6 w-11 items-center rounded-full transition-colors", systemPower ? 'bg-primary' : 'bg-muted')}
                    >
                        <span className={clsx("inline-block h-4 w-4 transform rounded-full bg-white transition-transform", systemPower ? 'translate-x-6' : 'translate-x-1')} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ControlPanel;
