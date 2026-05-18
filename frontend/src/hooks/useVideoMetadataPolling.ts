import { useEffect } from 'react';
import { selectVideoDetailDerivedMetadata, useVideoStore } from '../store/useVideoStore';

const VIDEO_METADATA_POLL_INTERVAL_MS = 4000;

/**
 * Polls video metadata while auxiliary media jobs can change VoiceFixer or reconstruction status.
 */
export function useVideoMetadataPolling(videoId?: string) {
    const { voiceFixerBusy, reconstructionBusy } = useVideoStore(selectVideoDetailDerivedMetadata);
    const fetchVideoMeta = useVideoStore((s) => s.fetchVideoMeta);

    useEffect(() => {
        if (!videoId || (!voiceFixerBusy && !reconstructionBusy)) return;
        const timer = window.setInterval(() => {
            void fetchVideoMeta(videoId);
        }, VIDEO_METADATA_POLL_INTERVAL_MS);
        return () => window.clearInterval(timer);
    }, [fetchVideoMeta, reconstructionBusy, videoId, voiceFixerBusy]);
}
