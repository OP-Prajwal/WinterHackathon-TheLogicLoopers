import React from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import classes from './MetricCard.module.css';

interface MetricCardProps {
    title: string;
    value: string | number;
    label?: string;
    status?: 'normal' | 'warning' | 'danger' | 'neutral';
    icon?: React.ReactNode;
    delay?: number;
}

export const MetricCard: React.FC<MetricCardProps> = ({
    title,
    value,
    label,
    status = 'neutral',
    icon,
    delay = 0
}) => {
    return (
        <motion.div
            className={clsx(classes.card, classes[status])}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay }}
        >
            <div className={classes.header}>
                <span className={classes.title}>{title}</span>
                {icon && <div className={classes.icon}>{icon}</div>}
            </div>
            <div className={classes.content}>
                <div className={classes.value}>{value}</div>
                {label && <div className={classes.label}>{label}</div>}
            </div>
        </motion.div>
    );
};
