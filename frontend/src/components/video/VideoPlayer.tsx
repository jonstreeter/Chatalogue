import { startTransition, useEffect, useRef } from 'react';
import YouTube from 'react-youtube';
import type { Video } from '../../types';
import { toApiUrl } from '../../lib/api';
import { AudioLines, Loader2, PlayCircle, Sparkles, type LucideIcon } from 'lucide-react';
import { usePlayerStore, type UnifiedPlayer } from '../../store/usePlayerStore';

export type UploadedPlaybackSource = 'original' | 'cleaned' | 'reconstructed';

type PlayerClockSnapshot = {
    mediaTime: number;
    wallTimeMs: number;
    playbackRate: number;
    playerState: number;
};

const PLAYER_UI_UPDATE_INTERVAL_MS = 120;
const PLAYER_UI_MIN_DELTA_SECONDS = 0.12;

export type VideoPlayerProps = {
    video: Video | null;
    containerClassName: string;
    isLocallyHostedMedia: boolean;
    isUploadedMedia: boolean;
    isTikTokMedia: boolean;
    localMediaPending: boolean;
    isUploadedAudio: boolean;
    requestedJumpTime: number;
    playbackRate: number;
    currentUploadedPlaybackSource: UploadedPlaybackSource;
    usingReconstructionForPlayback: boolean;
    reconstructionStatus: string;
    usingVoiceFixerForPlayback: boolean;
    voiceFixerApplyScope: string;
    hasVoiceFixerCleaned: boolean;
    hasReconstructionAudio: boolean;
    usingVoiceFixerForProcessing: boolean;
    episodeBusy: boolean;
    switchingReconstructionPlayback: boolean;
    onUploadedPlaybackSourceChange: (source: UploadedPlaybackSource) => void | Promise<void>;
    onPlayerClock?: (snapshot: PlayerClockSnapshot) => void;
};

export function VideoPlayer({
    video,
    containerClassName,
    isLocallyHostedMedia,
    isUploadedMedia,
    isTikTokMedia,
    localMediaPending,
    isUploadedAudio,
    requestedJumpTime,
    playbackRate,
    currentUploadedPlaybackSource,
    usingReconstructionForPlayback,
    reconstructionStatus,
    usingVoiceFixerForPlayback,
    voiceFixerApplyScope,
    hasVoiceFixerCleaned,
    hasReconstructionAudio,
    usingVoiceFixerForProcessing,
    episodeBusy,
    switchingReconstructionPlayback,
    onUploadedPlaybackSourceChange,
    onPlayerClock,
}: VideoPlayerProps) {
    const player = usePlayerStore((s) => s.player);
    const setPlayer = usePlayerStore((s) => s.setPlayer);
    const setCurrentTime = usePlayerStore((s) => s.setCurrentTime);
    const setPlaybackRate = usePlayerStore((s) => s.setPlaybackRate);
    const nativeMediaRef = useRef<HTMLMediaElement | null>(null);
    const nativeInitialSeekDoneRef = useRef(false);
    const youtubeInitialSeekDoneRef = useRef(false);
    const lastCurrentTimeUiUpdateMsRef = useRef(0);
    const lastPublishedCurrentTimeRef = useRef(0);
    const playerClockRef = useRef<PlayerClockSnapshot>({
        mediaTime: 0,
        wallTimeMs: 0,
        playbackRate: 1,
        playerState: -1,
    });
    const previousLocalMediaUrlRef = useRef('');
    const pendingNativeSourceRestoreRef = useRef<{ currentTime: number; wasPlaying: boolean; playbackRate: number } | null>(null);

    const localMediaUrl = (() => {
        if (!video || !isLocallyHostedMedia) return '';
        const params = new URLSearchParams({
            reconstruction_variant: usingReconstructionForPlayback ? 'reconstructed' : 'source',
            reconstruction_enabled: usingReconstructionForPlayback ? 'true' : 'false',
            reconstruction_status: reconstructionStatus || 'none',
            media_variant: usingVoiceFixerForPlayback ? 'clean' : 'original',
            media_scope: voiceFixerApplyScope,
            media_status: String(video.voicefixer_status || 'none'),
        });
        return `${toApiUrl(`/videos/${video.id}/media`)}?${params.toString()}`;
    })();

    useEffect(() => {
        nativeInitialSeekDoneRef.current = false;
        youtubeInitialSeekDoneRef.current = false;
        lastCurrentTimeUiUpdateMsRef.current = 0;
        lastPublishedCurrentTimeRef.current = 0;
        playerClockRef.current = {
            mediaTime: 0,
            wallTimeMs: 0,
            playbackRate: 1,
            playerState: -1,
        };
    }, [video?.id]);

    useEffect(() => {
        if (!localMediaUrl) {
            previousLocalMediaUrlRef.current = '';
            pendingNativeSourceRestoreRef.current = null;
            return;
        }

        if (previousLocalMediaUrlRef.current && previousLocalMediaUrlRef.current !== localMediaUrl) {
            const element = nativeMediaRef.current;
            pendingNativeSourceRestoreRef.current = {
                currentTime: Math.max(0, Number(element?.currentTime || 0)),
                wasPlaying: !!element && !element.paused && !element.ended,
                playbackRate: Math.max(0.75, Number(element?.playbackRate || playbackRate || 1)),
            };
        }

        previousLocalMediaUrlRef.current = localMediaUrl;
    }, [localMediaUrl, playbackRate]);

    const publishCurrentTime = (mediaTime: number, options?: { force?: boolean }) => {
        if (!Number.isFinite(mediaTime)) return;
        const now = performance.now();
        const force = !!options?.force;
        const lastUpdateMs = lastCurrentTimeUiUpdateMsRef.current;
        const lastPublished = lastPublishedCurrentTimeRef.current;

        if (!force) {
            if ((now - lastUpdateMs) < PLAYER_UI_UPDATE_INTERVAL_MS) return;
            if (Math.abs(mediaTime - lastPublished) < PLAYER_UI_MIN_DELTA_SECONDS) return;
        }

        lastCurrentTimeUiUpdateMsRef.current = now;
        lastPublishedCurrentTimeRef.current = mediaTime;
        startTransition(() => {
            setCurrentTime(prev => (Math.abs(prev - mediaTime) >= 0.01 ? mediaTime : prev));
        });
    };

    const publishPlayerClock = (snapshot: PlayerClockSnapshot) => {
        playerClockRef.current = snapshot;
        onPlayerClock?.(snapshot);
    };

    const buildNativePlayerAdapter = (element: HTMLMediaElement): UnifiedPlayer => ({
        getCurrentTime: () => Number(element.currentTime || 0),
        seekTo: (seconds: number) => {
            element.currentTime = Math.max(0, Number(seconds || 0));
        },
        playVideo: () => {
            void element.play().catch(() => { });
        },
        pauseVideo: () => {
            element.pause();
        },
        getPlaybackRate: () => Number(element.playbackRate || 1),
        setPlaybackRate: (rate: number) => {
            element.playbackRate = rate;
        },
        getPlayerState: () => {
            if (element.ended) return 0;
            return element.paused ? 2 : 1;
        },
    });

    const syncNativePlayerClock = (element: HTMLMediaElement) => {
        const nextRate = Number(element.playbackRate || 1);
        const nextState = element.ended ? 0 : (element.paused ? 2 : 1);
        const mediaTime = Number(element.currentTime || 0);
        const snapshot = {
            mediaTime,
            wallTimeMs: performance.now(),
            playbackRate: nextRate,
            playerState: nextState,
        };
        publishPlayerClock(snapshot);
        setPlaybackRate(nextRate);
        publishCurrentTime(mediaTime, { force: true });
    };

    const sampleYoutubePlayerClock = (ytPlayer: any, options?: { force?: boolean }) => {
        if (!ytPlayer) return;
        try {
            const mediaTimeRaw = ytPlayer.getCurrentTime?.();
            const stateRaw = ytPlayer.getPlayerState?.();
            const rateRaw = ytPlayer.getPlaybackRate?.();
            const mediaTime = Number(mediaTimeRaw);
            const state = Number(stateRaw);
            const rate = Number(rateRaw);
            if (!Number.isFinite(mediaTime)) return;

            const nextRate = Number.isFinite(rate) && rate > 0 ? rate : playerClockRef.current.playbackRate;
            const nextState = Number.isFinite(state) ? state : playerClockRef.current.playerState;
            const snapshot = {
                mediaTime,
                wallTimeMs: performance.now(),
                playbackRate: nextRate,
                playerState: nextState,
            };

            publishPlayerClock(snapshot);
            publishCurrentTime(mediaTime, { force: options?.force });
        } catch {
            // Ignore transient iframe/player API failures.
        }
    };

    const handleYoutubeReady = (event: any) => {
        setPlayer(event.target);
        sampleYoutubePlayerClock(event.target, { force: true });
        const rate = Number(event?.target?.getPlaybackRate?.()) || 1;
        setPlaybackRate(rate);
        if (!youtubeInitialSeekDoneRef.current && Number.isFinite(requestedJumpTime) && requestedJumpTime >= 0) {
            youtubeInitialSeekDoneRef.current = true;
            window.setTimeout(() => {
                try {
                    if (typeof event.target.seekTo === 'function') {
                        event.target.seekTo(requestedJumpTime, true);
                    }
                    // Do not autoplay when opening episode detail (including deep links).
                    if (typeof event.target.pauseVideo === 'function') {
                        event.target.pauseVideo();
                    }
                    sampleYoutubePlayerClock(event.target, { force: true });
                } catch (e) {
                    console.warn('Initial timestamp seek failed', e);
                    youtubeInitialSeekDoneRef.current = false;
                }
            }, 150);
        }
    };

    const handleYoutubeStateChange = (event: any) => {
        const state = Number(event?.data);
        if (!Number.isFinite(state)) return;
        let mediaTime = playerClockRef.current.mediaTime;
        try {
            const t = Number(event?.target?.getCurrentTime?.());
            if (Number.isFinite(t)) mediaTime = t;
        } catch {
            // Ignore and keep last known media time.
        }
        const snapshot = {
            ...playerClockRef.current,
            playerState: state,
            mediaTime,
            wallTimeMs: performance.now(),
        };
        publishPlayerClock(snapshot);
        publishCurrentTime(mediaTime, { force: true });
    };

    const handleYoutubePlaybackRateChange = (event: any) => {
        const rate = Number(event?.data ?? event?.target?.getPlaybackRate?.());
        if (!Number.isFinite(rate) || rate <= 0) return;
        let mediaTime = playerClockRef.current.mediaTime;
        try {
            const t = Number(event?.target?.getCurrentTime?.());
            if (Number.isFinite(t)) mediaTime = t;
        } catch {
            // Ignore and keep last known media time.
        }
        const snapshot = {
            ...playerClockRef.current,
            playbackRate: rate,
            mediaTime,
            wallTimeMs: performance.now(),
        };
        publishPlayerClock(snapshot);
        setPlaybackRate(rate);
        publishCurrentTime(mediaTime, { force: true });
    };

    useEffect(() => {
        if (!player || isLocallyHostedMedia) return;

        sampleYoutubePlayerClock(player, { force: true });
        // Use YouTube player time as the single source of truth.
        // This avoids drift/overshoot from extrapolation when iframe timing events stall.
        const pollId = window.setInterval(() => sampleYoutubePlayerClock(player), 33);

        return () => {
            window.clearInterval(pollId);
        };
    }, [isLocallyHostedMedia, player]);

    const handleNativeMediaRef = (element: HTMLMediaElement | null) => {
        nativeMediaRef.current = element;
    };

    const handlePlaybackRateChange = (rate: number) => {
        const nextRate = Number(rate);
        if (!Number.isFinite(nextRate) || nextRate <= 0 || !player) return;
        try {
            player.setPlaybackRate?.(nextRate);
            if (nativeMediaRef.current) {
                nativeMediaRef.current.playbackRate = nextRate;
                syncNativePlayerClock(nativeMediaRef.current);
            }
            setPlaybackRate(nextRate);
        } catch (e) {
            console.warn('Failed to change playback rate', e);
        }
    };

    const initializeNativeMedia = (element: HTMLMediaElement, options?: { skipInitialSeek?: boolean }) => {
        nativeMediaRef.current = element;
        setPlayer(buildNativePlayerAdapter(element));
        syncNativePlayerClock(element);
        if (!options?.skipInitialSeek && !nativeInitialSeekDoneRef.current && Number.isFinite(requestedJumpTime) && requestedJumpTime >= 0) {
            nativeInitialSeekDoneRef.current = true;
            window.setTimeout(() => {
                try {
                    element.currentTime = Math.max(0, requestedJumpTime);
                    element.pause();
                    syncNativePlayerClock(element);
                } catch (e) {
                    console.warn('Initial timestamp seek failed', e);
                    nativeInitialSeekDoneRef.current = false;
                }
            }, 120);
        }
    };

    const handleNativeMediaLoadedMetadata = (element: HTMLMediaElement) => {
        const pendingRestore = pendingNativeSourceRestoreRef.current;
        if (!pendingRestore) {
            initializeNativeMedia(element);
            return;
        }

        pendingNativeSourceRestoreRef.current = null;
        initializeNativeMedia(element, { skipInitialSeek: true });
        window.setTimeout(() => {
            try {
                element.currentTime = Math.max(0, pendingRestore.currentTime);
                element.playbackRate = pendingRestore.playbackRate;
                syncNativePlayerClock(element);
                if (pendingRestore.wasPlaying) {
                    const playAttempt = element.play();
                    if (playAttempt && typeof (playAttempt as Promise<void>).catch === 'function') {
                        void (playAttempt as Promise<void>).catch(() => { });
                    }
                }
            } catch (e) {
                console.warn('Playback source restore failed', e);
            }
        }, 120);
    };

    const renderUploadedPlaybackSourceSwitcher = () => {
        if (!isUploadedMedia) return null;

        const options: Array<{
            id: UploadedPlaybackSource;
            label: string;
            detail: string;
            icon: LucideIcon;
            available: boolean;
            activeClassName: string;
            inactiveClassName: string;
        }> = [
                {
                    id: 'original',
                    label: 'Original',
                    detail: 'Uploaded media',
                    icon: PlayCircle,
                    available: true,
                    activeClassName: 'border-slate-300 bg-slate-900 text-white shadow-sm',
                    inactiveClassName: 'border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50',
                },
                {
                    id: 'cleaned',
                    label: 'Cleanup',
                    detail: hasVoiceFixerCleaned ? 'VoiceFixer pass' : 'Run cleanup first',
                    icon: AudioLines,
                    available: hasVoiceFixerCleaned,
                    activeClassName: 'border-sky-200 bg-sky-600 text-white shadow-sm',
                    inactiveClassName: 'border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100',
                },
                {
                    id: 'reconstructed',
                    label: 'Rebuild',
                    detail: hasReconstructionAudio ? 'Conversation rebuild' : 'Run reconstruction first',
                    icon: Sparkles,
                    available: hasReconstructionAudio,
                    activeClassName: 'border-violet-200 bg-violet-600 text-white shadow-sm',
                    inactiveClassName: 'border-violet-200 bg-violet-50 text-violet-700 hover:bg-violet-100',
                },
            ];

        return (
            <div className="mt-2 rounded-xl border border-slate-200 bg-white px-3 py-3 text-xs shadow-sm">
                <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
                    <div className="min-w-0">
                        <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Playback Source</div>
                        <div className="mt-1 text-slate-700">
                            Switch the player between the original upload and generated variants while you listen.
                        </div>
                        <div className="mt-1 text-[11px] text-slate-500">
                            Processing stays on {usingVoiceFixerForProcessing ? 'the cleaned pass' : 'the original upload'}.
                        </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {options.map((option) => {
                            const Icon = option.icon;
                            const active = currentUploadedPlaybackSource === option.id;
                            return (
                                <button
                                    key={option.id}
                                    type="button"
                                    onClick={() => void onUploadedPlaybackSourceChange(option.id)}
                                    disabled={!option.available || episodeBusy || switchingReconstructionPlayback}
                                    title={!option.available ? option.detail : `Use ${option.label.toLowerCase()} audio for playback`}
                                    className={`inline-flex min-w-[122px] items-center gap-2 rounded-xl border px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${active ? option.activeClassName : option.inactiveClassName
                                        }`}
                                >
                                    {switchingReconstructionPlayback && active ? <Loader2 size={14} className="animate-spin" /> : <Icon size={14} />}
                                    <span className="min-w-0">
                                        <span className="block text-[11px] font-semibold uppercase tracking-wide">{option.label}</span>
                                        <span className={`block truncate text-[11px] ${active ? 'text-white/80' : 'text-slate-500'}`}>{option.detail}</span>
                                    </span>
                                </button>
                            );
                        })}
                    </div>
                </div>
            </div>
        );
    };

    const renderPlaybackRateControl = () => (
        <div className="mt-2">
            <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 shadow-sm">
                <span>
                    {isUploadedMedia
                        ? (isUploadedAudio ? 'Uploaded audio' : 'Uploaded video')
                        : isTikTokMedia
                            ? 'TikTok local media'
                            : 'YouTube player'}
                </span>
                <div className="flex items-center gap-2">
                    <span>Speed</span>
                    <select
                        value={String(playbackRate || 1)}
                        onChange={(e) => handlePlaybackRateChange(Number(e.target.value))}
                        className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                    >
                        {[0.75, 1, 1.25, 1.5, 1.75, 2].map(rate => (
                            <option key={rate} value={rate}>{rate}x</option>
                        ))}
                    </select>
                </div>
            </div>
            {renderUploadedPlaybackSourceSwitcher()}
        </div>
    );

    return (
        <>
            <div className={containerClassName}>
                {isLocallyHostedMedia ? (
                    localMediaPending ? (
                        <div className="flex h-full w-full items-center justify-center bg-[radial-gradient(circle_at_top,rgba(236,72,153,.22),transparent_45%),linear-gradient(135deg,#111827,#1f2937)] px-8 text-white">
                            <div className="max-w-lg rounded-2xl border border-white/10 bg-white/10 p-6 text-center backdrop-blur-sm">
                                <div className="text-sm font-semibold">Local TikTok media is not ready yet</div>
                                <div className="mt-2 text-xs leading-relaxed text-white/75">
                                    Playback switches to the native player after the TikTok file has been downloaded locally. Start processing or wait for the download stage to complete.
                                </div>
                            </div>
                        </div>
                    ) : isUploadedAudio ? (
                        <div className="flex h-full w-full items-center justify-center bg-[radial-gradient(circle_at_top,rgba(59,130,246,.24),transparent_45%),linear-gradient(135deg,#0f172a,#1e293b)] px-8">
                            <div className="w-full max-w-2xl rounded-2xl border border-white/10 bg-white/10 p-6 text-white backdrop-blur-sm">
                                <div className="mb-4 flex items-center gap-3">
                                    <AudioLines size={18} className="text-blue-200" />
                                    <div>
                                        <div className="text-sm font-semibold">Audio Episode</div>
                                        <div className="text-xs text-blue-100/80">{video?.title}</div>
                                    </div>
                                </div>
                                <audio
                                    ref={handleNativeMediaRef}
                                    src={localMediaUrl}
                                    controls
                                    preload="metadata"
                                    className="w-full"
                                    onLoadedMetadata={(e) => handleNativeMediaLoadedMetadata(e.currentTarget)}
                                    onTimeUpdate={(e) => syncNativePlayerClock(e.currentTarget)}
                                    onPlay={(e) => syncNativePlayerClock(e.currentTarget)}
                                    onPause={(e) => syncNativePlayerClock(e.currentTarget)}
                                    onRateChange={(e) => syncNativePlayerClock(e.currentTarget)}
                                    onEnded={(e) => syncNativePlayerClock(e.currentTarget)}
                                />
                            </div>
                        </div>
                    ) : (
                        <video
                            ref={handleNativeMediaRef}
                            src={localMediaUrl}
                            controls
                            preload="metadata"
                            className="h-full w-full bg-black"
                            onLoadedMetadata={(e) => handleNativeMediaLoadedMetadata(e.currentTarget)}
                            onTimeUpdate={(e) => syncNativePlayerClock(e.currentTarget)}
                            onPlay={(e) => syncNativePlayerClock(e.currentTarget)}
                            onPause={(e) => syncNativePlayerClock(e.currentTarget)}
                            onRateChange={(e) => syncNativePlayerClock(e.currentTarget)}
                            onEnded={(e) => syncNativePlayerClock(e.currentTarget)}
                        />
                    )
                ) : (
                    <YouTube
                        videoId={video?.youtube_id || ''}
                        className="w-full h-full"
                        iframeClassName="w-full h-full"
                        onReady={handleYoutubeReady}
                        onStateChange={handleYoutubeStateChange}
                        onPlaybackRateChange={handleYoutubePlaybackRateChange}
                        opts={{
                            height: '100%',
                            width: '100%',
                            playerVars: {
                                autoplay: 0,
                                modestbranding: 1,
                                rel: 0,
                                ...(Number.isFinite(requestedJumpTime) && requestedJumpTime >= 0
                                    ? { start: Math.floor(requestedJumpTime) }
                                    : {}),
                            },
                        }}
                    />
                )}
            </div>
            {renderPlaybackRateControl()}
        </>
    );
}
