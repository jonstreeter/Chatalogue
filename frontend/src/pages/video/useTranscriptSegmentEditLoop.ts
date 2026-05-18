import { useEffect } from 'react';
import { usePlayerStore } from '../../store/usePlayerStore';
import { useTranscriptStore } from '../../store/useTranscriptStore';
import { useVideoStore } from '../../store/useVideoStore';

export function useTranscriptSegmentEditLoop() {
    const player = usePlayerStore((s) => s.player);
    const segments = useVideoStore((s) => s.segments);
    const editingSegmentId = useTranscriptStore((s) => s.editingSegmentId);
    const editingLoopSegment = useTranscriptStore((s) => s.editingLoopSegment);

    useEffect(() => {
        if (!editingSegmentId || !editingLoopSegment || !player) return;
        const seg = segments.find((s) => s.id === editingSegmentId);
        if (!seg) return;

        try {
            player.seekTo?.(seg.start_time, true);
            player.playVideo?.();
        } catch {
            // ignore
        }

        const loopId = window.setInterval(() => {
            try {
                const t = Number(player.getCurrentTime?.());
                if (!Number.isFinite(t)) return;
                if (t >= Math.max(seg.start_time, seg.end_time - 0.12)) {
                    player.seekTo?.(seg.start_time, true);
                    player.playVideo?.();
                }
            } catch {
                // ignore transient iframe failures
            }
        }, 100);

        return () => window.clearInterval(loopId);
    }, [editingSegmentId, editingLoopSegment, player, segments]);
}
