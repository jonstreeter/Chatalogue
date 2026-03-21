
import { HardDrive, FileText, Users } from 'lucide-react';

interface VideoStatusBadgeProps {
    status: string;
    processed: boolean;
    accessRestricted?: boolean;
    className?: string;
}

export function VideoStatusBadge({ status, processed, accessRestricted = false, className = '' }: VideoStatusBadgeProps) {
    const s = status.toLowerCase();

    // Helper to determine state of each step: 'pending' | 'active' | 'completed' | 'failed'
    const getStepState = (step: 'download' | 'transcribe' | 'diarize') => {
        if (accessRestricted) return 'restricted';
        if (processed) return 'completed';
        if (s === 'failed') return 'failed'; // Simplified: if failed, show all as failed or just stop? Let's just return failed for now.

        const order = ['pending', 'queued', 'downloading', 'downloaded', 'transcribing', 'diarizing', 'completed'];
        const stepIndex = { download: 2, transcribe: 4, diarize: 5 };
        const currentIndex = order.indexOf(s);

        // Map current status to index
        // pending=0, queued=1, downloading=2, downloaded=3, transcribing=4, diarizing=5, completed=6

        const targetIndex = stepIndex[step];

        if (currentIndex > targetIndex) return 'completed';
        if (currentIndex === targetIndex) return 'active';
        return 'pending';
    };

    const renderIcon = (Icon: any, step: 'download' | 'transcribe' | 'diarize', label: string) => {
        const state = getStepState(step);
        let colorClass = 'text-slate-300'; // pending
        let animate = false;

        if (state === 'completed') colorClass = 'text-green-500';
        else if (state === 'active') {
            colorClass = 'text-blue-500';
            animate = true;
        }
        else if (state === 'restricted') colorClass = 'text-slate-400';
        else if (state === 'failed') colorClass = 'text-red-400';

        return (
            <div className={`p-1 rounded-full bg-white/50 ${animate ? 'animate-pulse' : ''}`} title={`${label}: ${state}`}>
                <Icon size={14} className={colorClass} />
            </div>
        );
    };

    return (
        <div className={`flex items-center gap-1 ${className}`}>
            {renderIcon(HardDrive, 'download', 'Download')}
            {renderIcon(FileText, 'transcribe', 'Transcription')}
            {renderIcon(Users, 'diarize', 'Diarization')}
        </div>
    );
}
