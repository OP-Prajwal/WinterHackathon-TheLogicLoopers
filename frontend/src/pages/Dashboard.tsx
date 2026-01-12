import React from 'react';
import { Activity, ShieldCheck, AlertTriangle, Database } from 'lucide-react';
import MetricCard from '../components/metrics/MetricCard';
import EffectiveRankChart from '../components/metrics/EffectiveRankChart';
import ControlPanel from '../components/dashboard/ControlPanel';
import EventLog from '../components/dashboard/EventLog';
import PurificationPanel from '../components/dashboard/PurificationPanel';
import ManualTest from '../components/dashboard/ManualTest';
import { fetchMetrics, fetchRankHistory, type Metrics, type RankData } from '../services/dashboardService';

const Dashboard: React.FC = () => {
    const [metrics, setMetrics] = React.useState<Metrics | null>(null);
    const [chartData, setChartData] = React.useState<RankData[]>([]);

    React.useEffect(() => {
        const loadData = async () => {
            try {
                const [m, c] = await Promise.all([fetchMetrics(), fetchRankHistory()]);
                setMetrics(m);
                setChartData(c);
            } catch (e) {
                console.error(e);
            }
        };
        loadData();
    }, []);

    if (!metrics) return <div className="p-8">Loading...</div>;

    return (
        <div className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                    label="Total Scans"
                    value={metrics.totalScans.toLocaleString()}
                    change="+12% from last month"
                    trend="up"
                    icon={Activity}
                />
                <MetricCard
                    label="Security Score"
                    value={`${metrics.securityScore}%`}
                    change="+2.1% from last week"
                    trend="up"
                    icon={ShieldCheck}
                />
                <MetricCard
                    label="Threats Detected"
                    value={metrics.threatsDetected}
                    change="-5% from last month"
                    trend="down"
                    icon={AlertTriangle}
                />
                <MetricCard
                    label="Database Size"
                    value={metrics.databaseSize}
                    change="+0.5% from last month"
                    trend="neutral"
                    icon={Database}
                />
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
                <div className="col-span-4">
                    <EffectiveRankChart data={chartData} />
                </div>
                <div className="col-span-3 bg-card rounded-lg border border-border p-6 text-card-foreground">
                    <h3 className="font-semibold text-lg mb-4">Recent Alerts</h3>
                    <div className="space-y-4">
                        <div className="flex items-center">
                            <div className="ml-4 space-y-1">
                                <p className="text-sm font-medium leading-none">Suspicious IP Blocked</p>
                                <p className="text-xs text-muted-foreground">10 minutes ago</p>
                            </div>
                            <div className="ml-auto font-medium text-destructive">High</div>
                        </div>
                        <div className="flex items-center">
                            <div className="ml-4 space-y-1">
                                <p className="text-sm font-medium leading-none">New User Admin</p>
                                <p className="text-xs text-muted-foreground">1 hour ago</p>
                            </div>
                            <div className="ml-auto font-medium text-green-500">Low</div>
                        </div>
                        <div className="flex items-center">
                            <div className="ml-4 space-y-1">
                                <p className="text-sm font-medium leading-none">System Update</p>
                                <p className="text-xs text-muted-foreground">2 hours ago</p>
                            </div>
                            <div className="ml-auto font-medium text-blue-500">Info</div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
                <ControlPanel />
                <EventLog />
            </div>


            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <div className="lg:col-span-1">
                    <PurificationPanel />
                </div>
                <div className="lg:col-span-2">
                    <ManualTest />
                </div>
            </div>
        </div >
    );
};

export default Dashboard;
