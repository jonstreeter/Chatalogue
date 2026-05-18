import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import type { EpisodeChatCitation, TranscriptSegment } from '../../types';
import { usePlayerStore } from '../../store/usePlayerStore';

export type VideoSidebarTab = 'transcript' | 'optimize' | 'clips' | 'speakers' | 'cleanup' | 'reconstruction' | 'clone' | 'chat' | 'youtube';

type PlayerClockSnapshot = {
    mediaTime: number;
    wallTimeMs: number;
    playbackRate: number;
    playerState: number;
};

type UseVideoSeekOptions = {
    videoId?: string;
    activeTab: VideoSidebarTab;
    setActiveTab: (tab: VideoSidebarTab) => void;
    segments: TranscriptSegment[];
    requestedJumpTime: number;
    requestedSegmentId: number;
};

const TRANSCRIPT_SEGMENT_TRAIL_SECONDS = 0.05;

export const useVideoSeek = ({
    videoId,
    activeTab,
    setActiveTab,
    segments,
    requestedJumpTime,
    requestedSegmentId,
}: UseVideoSeekOptions) => {
    const navigate = useNavigate();
    const player = usePlayerStore((s) => s.player);
    const initialJumpDoneRef = useRef(false);
    const playerClockRef = useRef<PlayerClockSnapshot>({
        mediaTime: 0,
        wallTimeMs: 0,
        playbackRate: 1,
        playerState: -1,
    });

    useEffect(() => {
        initialJumpDoneRef.current = false;
    }, [videoId]);

    const handleSeek = (time: number) => {
        if (!player) return;
        try {
            if (typeof player.seekTo === 'function') {
                player.seekTo(time, true);
            }
            if (typeof player.playVideo === 'function') {
                player.playVideo();
            }
            const rate = Number(player?.getPlaybackRate?.());
            playerClockRef.current = {
                ...playerClockRef.current,
                mediaTime: time,
                wallTimeMs: performance.now(),
                playerState: 1,
                playbackRate: Number.isFinite(rate) && rate > 0 ? rate : playerClockRef.current.playbackRate,
            };
            usePlayerStore.getState().setCurrentTime(time);
        } catch (e) {
            console.warn('Failed to seek video player', e);
        }
    };

    const syncPlayerClockSnapshot = (snapshot: PlayerClockSnapshot) => {
        playerClockRef.current = {
            mediaTime: snapshot.mediaTime,
            wallTimeMs: snapshot.wallTimeMs,
            playbackRate: snapshot.playbackRate,
            playerState: snapshot.playerState,
        };
    };

    const scrollTranscriptToSegment = (segmentId: number) => {
        if (!segmentId) return;

        const doScroll = () => {
            const el = document.getElementById(`seg-${segmentId}`);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        };

        if (activeTab !== 'transcript') {
            setActiveTab('transcript');
            window.setTimeout(doScroll, 80);
        } else {
            doScroll();
        }
    };

    const scrollTranscriptToTime = (time: number) => {
        if (segments.length === 0) return;

        const target =
            segments.find(s => time >= s.start_time && time < s.end_time + TRANSCRIPT_SEGMENT_TRAIL_SECONDS) ||
            segments.find(s => s.start_time >= time) ||
            segments[segments.length - 1];

        if (!target) return;

        const doScroll = () => {
            const el = document.getElementById(`seg-${target.id}`);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        };

        if (activeTab !== 'transcript') {
            setActiveTab('transcript');
            window.setTimeout(doScroll, 80);
        } else {
            doScroll();
        }
    };

    useEffect(() => {
        if (initialJumpDoneRef.current) return;
        if (segments.length === 0) return;
        const hasRequestedSegment = Number.isInteger(requestedSegmentId) && requestedSegmentId > 0;
        const hasRequestedJumpTime = Number.isFinite(requestedJumpTime) && requestedJumpTime >= 0;
        if (!hasRequestedSegment && !hasRequestedJumpTime) return;

        initialJumpDoneRef.current = true;
        const timer = window.setTimeout(() => {
            try {
                if (hasRequestedSegment && segments.some((seg) => seg.id === requestedSegmentId)) {
                    scrollTranscriptToSegment(requestedSegmentId);
                } else if (hasRequestedJumpTime) {
                    scrollTranscriptToTime(requestedJumpTime);
                }
            } catch (e) {
                console.warn('Initial transcript scroll failed', e);
            }
        }, 80);

        return () => window.clearTimeout(timer);
    }, [requestedJumpTime, requestedSegmentId, segments]);

    const handleCitationClick = (citation: EpisodeChatCitation) => {
        const citationVideoId = Number(citation.video_id || 0);
        const jumpTime = Number(citation.start_time || 0);
        const primarySegmentId = Array.isArray(citation.segment_ids) ? Number(citation.segment_ids[0] || 0) : 0;
        if (citationVideoId > 0 && citationVideoId !== Number(videoId)) {
            const params = new URLSearchParams();
            params.set('tab', 'chat');
            if (Number.isFinite(jumpTime) && jumpTime >= 0) {
                params.set('t', String(Math.floor(jumpTime)));
            }
            if (primarySegmentId > 0) {
                params.set('segment_id', String(primarySegmentId));
            }
            navigate(`/video/${citationVideoId}?${params.toString()}`);
            return;
        }
        handleSeek(jumpTime);
        if (primarySegmentId > 0) {
            scrollTranscriptToSegment(primarySegmentId);
        } else {
            scrollTranscriptToTime(jumpTime);
        }
    };

    return {
        handleSeek,
        handleCitationClick,
        scrollTranscriptToSegment,
        scrollTranscriptToTime,
        syncPlayerClockSnapshot,
    };
};

