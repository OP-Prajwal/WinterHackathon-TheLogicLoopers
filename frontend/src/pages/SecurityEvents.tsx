import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { API_BASE } from '../services/api';

interface ScanRecord {
    _id: string;
    scan_id: string;
    filename: string;
    timestamp: string;
    poison_count: number;
    safe_count: number;
    total_rows: number;
}

export function SecurityEvents() {
    const [scans, setScans] = useState<ScanRecord[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchScans();
    }, []);

    const fetchScans = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch(`${API_BASE}/api/scans`, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
            });
            if (response.ok) {
                const data = await response.json();
                // Reverse to show oldest to newest in chart
                setScans(data);
            } else {
                console.error("Failed to fetch scans:", response.statusText);
            }
        } catch (error) {
            console.error('Failed to fetch scans', error);
        } finally {
            setLoading(false);
        }
    };

    // Prepare chart data (oldest first)
    const chartData = [...scans].reverse().map(s => ({
        name: new Date(s.timestamp).toLocaleTimeString(),
        poison: s.poison_count,
        total: s.total_rows
    }));

    return (
        <div className="p-6">
            <h1 className="text-3xl font-bold text-white mb-6">Security Event Log</h1>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                {/* Metrics Cards */}
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <h3 className="text-gray-400 text-sm">Total Scans</h3>
                    <p className="text-2xl font-bold text-white">{scans.length}</p>
                </div>
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <h3 className="text-gray-400 text-sm">Threats Detected</h3>
                    <p className="text-2xl font-bold text-red-500">
                        {scans.reduce((acc, curr) => acc + curr.poison_count, 0)}
                    </p>
                </div>
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <h3 className="text-gray-400 text-sm">System Health</h3>
                    <p className="text-2xl font-bold text-green-500">Stable</p>
                </div>

                {/* Chart Section - Spans full width on mobile, 2 cols on wide */}
                <div className="lg:col-span-3 bg-gray-800 p-4 rounded-lg border border-gray-700 h-64">
                    <h3 className="text-gray-400 text-sm mb-4">Threat Detection Trend (Poison Count)</h3>
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                            <XAxis dataKey="name" stroke="#9CA3AF" style={{ fontSize: '10px' }} />
                            <YAxis stroke="#9CA3AF" style={{ fontSize: '10px' }} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', color: '#F3F4F6' }}
                                itemStyle={{ color: '#F3F4F6' }}
                            />
                            <Line type="monotone" dataKey="poison" stroke="#EF4444" strokeWidth={2} dot={{ fill: '#EF4444' }} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
                <table className="w-full text-left">
                    <thead className="bg-gray-900 text-gray-400 uppercase text-xs">
                        <tr>
                            <th className="p-4">Timestamp</th>
                            <th className="p-4">Scan ID</th>
                            <th className="p-4">Filename</th>
                            <th className="p-4">Status</th>
                            <th className="p-4">Details</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-700 text-gray-300">
                        {scans.map((scan) => (
                            <tr key={scan._id} className="hover:bg-gray-750 transition-colors">
                                <td className="p-4">{new Date(scan.timestamp).toLocaleString()}</td>
                                <td className="p-4 font-mono text-sm text-blue-400">{scan.scan_id}</td>
                                <td className="p-4">{scan.filename}</td>
                                <td className="p-4">
                                    {scan.poison_count > 0 ? (
                                        <span className="bg-red-900/50 text-red-300 px-2 py-1 rounded text-xs border border-red-800">
                                            POISON DETECTED
                                        </span>
                                    ) : (
                                        <span className="bg-green-900/50 text-green-300 px-2 py-1 rounded text-xs border border-green-800">
                                            SAFE
                                        </span>
                                    )}
                                </td>
                                <td className="p-4">
                                    Found {scan.poison_count} anomalies in {scan.total_rows} rows
                                </td>
                            </tr>
                        ))}
                        {scans.length === 0 && !loading && (
                            <tr>
                                <td colSpan={5} className="p-8 text-center text-gray-500">No security events found.</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
