import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const data = [
    { name: 'Jan', rank: 400 },
    { name: 'Feb', rank: 300 },
    { name: 'Mar', rank: 200 },
    { name: 'Apr', rank: 278 },
    { name: 'May', rank: 189 },
    { name: 'Jun', rank: 239 },
    { name: 'Jul', rank: 349 },
];

const EffectiveRankChart: React.FC = () => {
    return (
        <div className="w-full h-[300px] bg-card p-4 rounded-lg border border-border">
            <h3 className="text-lg font-semibold mb-4">Effective Rank History</h3>
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                    <Tooltip
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                        itemStyle={{ color: 'hsl(var(--foreground))' }}
                    />
                    <Line type="monotone" dataKey="rank" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default EffectiveRankChart;
