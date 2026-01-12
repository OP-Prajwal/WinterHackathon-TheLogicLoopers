import React from 'react';
import { type LucideIcon } from 'lucide-react';
import clsx from 'clsx';

interface MetricCardProps {
    label: string;
    value: string | number;
    icon: LucideIcon;
    change?: string;
    trend?: 'up' | 'down' | 'neutral';
}

const MetricCard: React.FC<MetricCardProps> = ({ label, value, icon: Icon, change, trend }) => {
    return (
        <div className="bg-card text-card-foreground p-6 rounded-lg border border-border shadow-sm">
            <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-medium text-muted-foreground">{label}</span>
                <div className="p-2 bg-primary/10 rounded-full">
                    <Icon size={20} className="text-primary" />
                </div>
            </div>
            <div className="flex items-baseline gap-2">
                <h3 className="text-3xl font-bold">{value}</h3>
            </div>
            {change && (
                <p className={clsx("text-xs mt-1", {
                    'text-green-500': trend === 'up',
                    'text-red-500': trend === 'down',
                    'text-muted-foreground': trend === 'neutral'
                })}>
                    {change}
                </p>
            )}
        </div>
    );
};

export default MetricCard;
