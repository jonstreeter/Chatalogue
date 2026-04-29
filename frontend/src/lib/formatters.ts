export function formatTime(seconds: number): string {
    const total = Math.max(0, Math.floor(seconds));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h > 0
        ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
        : `${m}:${s.toString().padStart(2, '0')}`;
}

export function formatViewMetric(value?: number | null, fractionDigits = 0): string {
    if (value == null || Number.isNaN(value)) return 'Unknown';
    return new Intl.NumberFormat(undefined, {
        maximumFractionDigits: fractionDigits,
        minimumFractionDigits: fractionDigits > 0 ? fractionDigits : 0,
    }).format(value);
}
