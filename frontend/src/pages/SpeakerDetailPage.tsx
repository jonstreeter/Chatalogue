import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import YouTube from 'react-youtube';
import api from '../lib/api';
import { toApiUrl } from '../lib/api';
import type { Speaker, SpeakerEpisodeAppearance, SpeakerVoiceProfile } from '../types';
import { ArrowLeft, Loader2, Mic2, Calendar, Hash, Trash2, AudioLines, Film, Pencil, Play, GitMerge, Search, PlusCircle, RefreshCw, X } from 'lucide-react';
import { SpeakerModal } from '../components/SpeakerModal';

const PROFILE_PREVIEW_STOP_EARLY_SECONDS = 0.35;

function formatDuration(seconds: number): string {
    const total = Math.max(0, Math.round(seconds));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    return `${m}:${s.toString().padStart(2, '0')}`;
}

export function SpeakerDetailPage() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();

    const [speaker, setSpeaker] = useState<Speaker | null>(null);
    const [appearances, setAppearances] = useState<SpeakerEpisodeAppearance[]>([]);
    const [profiles, setProfiles] = useState<SpeakerVoiceProfile[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [deletingProfileId, setDeletingProfileId] = useState<number | null>(null);
    const [editingSpeaker, setEditingSpeaker] = useState<Speaker | null>(null);
    const [previewProfile, setPreviewProfile] = useState<SpeakerVoiceProfile | null>(null);
    const [previewPlayer, setPreviewPlayer] = useState<any>(null);
    const [movingProfile, setMovingProfile] = useState<SpeakerVoiceProfile | null>(null);
    const [moveTargets, setMoveTargets] = useState<Speaker[]>([]);
    const [loadingMoveTargets, setLoadingMoveTargets] = useState(false);
    const [moveSearch, setMoveSearch] = useState('');
    const [movingProfileSubmit, setMovingProfileSubmit] = useState(false);
    const [newMoveSpeakerName, setNewMoveSpeakerName] = useState('');
    const [reassigningProfileId, setReassigningProfileId] = useState<number | null>(null);

    const stopPreviewPlayback = () => {
        try {
            previewPlayer?.pauseVideo?.();
            previewPlayer?.stopVideo?.();
        } catch {
            // Ignore transient iframe/player teardown errors.
        }
    };

    const closeProfilePreview = () => {
        stopPreviewPlayback();
        setPreviewPlayer(null);
        setPreviewProfile(null);
    };

    const fetchData = async () => {
        if (!id) return;
        setLoading(true);
        setError(null);
        try {
            const [speakerRes, appearancesRes, profilesRes] = await Promise.all([
                api.get<Speaker>(`/speakers/${id}`),
                api.get<SpeakerEpisodeAppearance[]>(`/speakers/${id}/appearances`),
                api.get<SpeakerVoiceProfile[]>(`/speakers/${id}/profiles`),
            ]);
            setSpeaker(speakerRes.data);
            setAppearances(Array.isArray(appearancesRes.data) ? appearancesRes.data : []);
            setProfiles(Array.isArray(profilesRes.data) ? profilesRes.data : []);
        } catch (e: any) {
            console.error('Failed to load speaker detail', e);
            setError(e?.response?.data?.detail || 'Failed to load speaker details');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void fetchData();
    }, [id]);

    const handleDeleteProfile = async (profile: SpeakerVoiceProfile) => {
        if (!speaker) return;
        const sourceLabel = profile.source_video_title || `profile #${profile.id}`;
        if (!confirm(`Remove voice profile from "${speaker.name}"?\n\nSource: ${sourceLabel}\n\nThis affects future speaker matching and diarization, but does not rewrite existing transcripts.`)) {
            return;
        }

        setDeletingProfileId(profile.id);
        try {
            const res = await api.delete(`/speakers/${speaker.id}/profiles/${profile.id}`);
            const remainingProfiles = Number(res.data?.remaining_profiles);
            setProfiles(prev => prev.filter(p => p.id !== profile.id));
            setPreviewProfile(prev => prev?.id === profile.id ? null : prev);
            if (previewProfile?.id === profile.id) {
                closeProfilePreview();
            }
            setSpeaker(prev => prev ? { ...prev, embedding_count: Number.isFinite(remainingProfiles) ? remainingProfiles : Math.max((prev.embedding_count || 1) - 1, 0) } : prev);
        } catch (e: any) {
            console.error('Failed to delete voice profile', e);
            alert(e?.response?.data?.detail || 'Failed to remove voice profile');
        } finally {
            setDeletingProfileId(null);
        }
    };

    const openMoveProfileDialog = async (profile: SpeakerVoiceProfile) => {
        if (!speaker) return;
        setMovingProfile(profile);
        setMoveSearch('');
        setNewMoveSpeakerName('');
        setMoveTargets([]);
        setLoadingMoveTargets(true);
        try {
            const res = await api.get<Speaker[]>('/speakers', {
                params: { channel_id: speaker.channel_id, limit: 5000 }
            });
            const items = Array.isArray(res.data) ? res.data : [];
            setMoveTargets(items.filter(s => s.id !== speaker.id));
        } catch (e) {
            console.error('Failed to load move targets', e);
            alert('Failed to load speakers for move target selection');
            setMovingProfile(null);
        } finally {
            setLoadingMoveTargets(false);
        }
    };

    const handleMoveProfileToExisting = async (targetSpeaker: Speaker) => {
        if (!speaker || !movingProfile) return;
        if (!confirm(`Move profile #${movingProfile.id} from "${speaker.name}" to "${targetSpeaker.name}"?`)) return;

        setMovingProfileSubmit(true);
        try {
            await api.post(`/speakers/${speaker.id}/profiles/${movingProfile.id}/move`, {
                target_speaker_id: targetSpeaker.id,
            });
            if (previewProfile?.id === movingProfile.id) {
                closeProfilePreview();
            }
            setMovingProfile(null);
            await fetchData();
        } catch (e: any) {
            console.error('Failed to move profile', e);
            alert(e?.response?.data?.detail || 'Failed to move profile');
        } finally {
            setMovingProfileSubmit(false);
        }
    };

    const handleMoveProfileToNewSpeaker = async () => {
        if (!speaker || !movingProfile) return;
        const trimmed = newMoveSpeakerName.trim();
        if (!trimmed) return;
        if (!confirm(`Create a new speaker "${trimmed}" and move profile #${movingProfile.id} into it?`)) return;

        setMovingProfileSubmit(true);
        try {
            const res = await api.post(`/speakers/${speaker.id}/profiles/${movingProfile.id}/move`, {
                new_speaker_name: trimmed,
            });
            const createdTargetId = Number(res.data?.target_speaker_id);
            if (previewProfile?.id === movingProfile.id) {
                closeProfilePreview();
            }
            setMovingProfile(null);
            await fetchData();
            if (Number.isFinite(createdTargetId) && createdTargetId > 0) {
                navigate(`/speakers/${createdTargetId}`);
            }
        } catch (e: any) {
            console.error('Failed to create/move profile', e);
            alert(e?.response?.data?.detail || 'Failed to move profile');
        } finally {
            setMovingProfileSubmit(false);
        }
    };

    const canPreviewProfile = (profile: SpeakerVoiceProfile) =>
        !!profile.source_video_id && profile.sample_start_time != null;

    const handleBulkReassignProfileSegments = async (profile: SpeakerVoiceProfile) => {
        if (!speaker) return;
        if (!confirm(`Reassign all segments matched to profile #${profile.id} to "${speaker.name}" across this channel?`)) return;
        setReassigningProfileId(profile.id);
        try {
            const res = await api.post(`/profiles/${profile.id}/reassign-segments`);
            const updated = Number(res.data?.updated_segments ?? 0);
            const total = Number(res.data?.matched_segments ?? 0);
            alert(`Bulk reassignment complete: ${updated} of ${total} matched segments reassigned.`);
        } catch (e: any) {
            console.error('Failed to bulk reassign profile segments', e);
            alert(e?.response?.data?.detail || 'Failed to bulk reassign matched segments');
        } finally {
            setReassigningProfileId(null);
        }
    };

    const handleProfilePreviewReady = (event: any) => {
        setPreviewPlayer(event.target);
        if (previewProfile?.sample_start_time != null) {
            try {
                event.target.seekTo(previewProfile.sample_start_time, true);
                event.target.playVideo?.();
            } catch (e) {
                console.warn('Failed to start profile preview', e);
            }
        }
    };

    const handleNativeProfilePreviewLoaded = (event: React.SyntheticEvent<HTMLMediaElement>) => {
        const media = event.currentTarget;
        const controller = {
            seekTo: (seconds: number) => {
                media.currentTime = Math.max(0, seconds || 0);
            },
            playVideo: async () => {
                try {
                    await media.play();
                } catch {
                    // Ignore autoplay restrictions for preview clips.
                }
            },
            pauseVideo: () => media.pause(),
            stopVideo: () => {
                media.pause();
                media.currentTime = 0;
            },
            getCurrentTime: () => media.currentTime || 0,
        };
        setPreviewPlayer(controller);
        if (previewProfile?.sample_start_time != null) {
            controller.seekTo(previewProfile.sample_start_time);
            void controller.playVideo();
        }
    };

    useEffect(() => {
        if (!previewPlayer || !previewProfile || previewProfile.sample_end_time == null || previewProfile.sample_start_time == null) return;
        const interval = setInterval(() => {
            try {
                const t = previewPlayer.getCurrentTime?.();
                const stopAt = Math.max(previewProfile.sample_start_time!, previewProfile.sample_end_time! - PROFILE_PREVIEW_STOP_EARLY_SECONDS);
                if (typeof t === 'number' && t >= stopAt) {
                    previewPlayer.pauseVideo?.();
                }
            } catch {
                // Ignore transient player errors while iframe reinitializes.
            }
        }, 200);
        return () => clearInterval(interval);
    }, [previewPlayer, previewProfile?.id, previewProfile?.sample_start_time, previewProfile?.sample_end_time]);

    useEffect(() => {
        return () => {
            try {
                previewPlayer?.pauseVideo?.();
                previewPlayer?.stopVideo?.();
            } catch {
                // Ignore transient iframe/player teardown errors.
            }
        };
    }, [previewPlayer]);

    const filteredMoveTargets = moveTargets.filter(s =>
        s.name.toLowerCase().includes(moveSearch.toLowerCase())
    );

    if (loading) {
        return (
            <div className="py-20 flex flex-col items-center justify-center text-slate-400 gap-4">
                <Loader2 className="w-8 h-8 animate-spin" />
                <p>Loading speaker details...</p>
            </div>
        );
    }

    if (error || !speaker) {
        return (
            <div className="space-y-4">
                <button onClick={() => navigate(-1)} className="inline-flex items-center gap-2 text-slate-500 hover:text-slate-700 text-sm">
                    <ArrowLeft size={16} /> Back
                </button>
                <div className="glass-panel rounded-xl p-8 text-center text-red-600">{error || 'Speaker not found'}</div>
            </div>
        );
    }

    const previewMediaSourceType = String(previewProfile?.source_video_media_source_type || 'youtube').toLowerCase();
    const previewUsesYoutube = previewMediaSourceType === 'youtube';
    const previewUsesLocalMedia = previewMediaSourceType === 'upload' || previewMediaSourceType === 'tiktok';
    const previewIsAudioOnly = previewUsesLocalMedia && String(previewProfile?.source_video_media_kind || '').toLowerCase() === 'audio';
    const previewMediaUrl = previewProfile?.source_video_id ? toApiUrl(`/videos/${previewProfile.source_video_id}/media`) : '';

    return (
        <div className="space-y-6 pb-20">
            <div className="flex items-center justify-between gap-4">
                <button onClick={() => navigate(-1)} className="inline-flex items-center gap-2 text-slate-500 hover:text-slate-700 text-sm">
                    <ArrowLeft size={16} /> Back
                </button>
                {speaker.channel_id && (
                    <Link
                        to={`/channel/${speaker.channel_id}/speakers`}
                        className="text-xs text-slate-500 hover:text-slate-700 underline decoration-slate-300 underline-offset-2"
                    >
                        Back to Channel Speakers
                    </Link>
                )}
            </div>

            <div className="glass-panel rounded-2xl p-5">
                <div className="flex items-start gap-4">
                    <div className="w-16 h-16 rounded-full bg-slate-100 overflow-hidden border border-slate-200 shrink-0">
                        {speaker.thumbnail_path ? (
                            <img src={toApiUrl(speaker.thumbnail_path)} alt={speaker.name} className="w-full h-full object-cover" />
                        ) : (
                            <div className="w-full h-full flex items-center justify-center text-slate-300">
                                <Mic2 size={24} />
                            </div>
                        )}
                    </div>
                    <div className="min-w-0 flex-1">
                        <div className="flex items-start justify-between gap-3">
                            <h1 className="text-2xl font-bold text-slate-800 truncate">{speaker.name}</h1>
                            <button
                                type="button"
                                onClick={() => setEditingSpeaker(speaker)}
                                className="shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-600"
                                title="Edit speaker"
                            >
                                <Pencil size={13} />
                                Edit
                            </button>
                        </div>
                        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                            <span className="px-2 py-1 rounded-full bg-slate-100 border border-slate-200">
                                Speaking time: {formatDuration(speaker.total_speaking_time)}
                            </span>
                            <span className="px-2 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-700">
                                Voice profiles: {profiles.length}
                            </span>
                            <span className="px-2 py-1 rounded-full bg-slate-100 border border-slate-200">
                                Episodes: {appearances.length}
                            </span>
                            {speaker.is_extra && (
                                <span className="px-2 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-700">Extra</span>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
                <section className="space-y-3 min-w-0">
                    <div className="flex items-center gap-2 text-slate-700">
                        <Film size={16} />
                        <h2 className="text-lg font-semibold">Episodes This Speaker Appears In</h2>
                    </div>
                    <div className="glass-panel rounded-xl border border-slate-200/60 overflow-hidden">
                        {appearances.length === 0 ? (
                            <div className="p-6 text-center text-slate-400 text-sm">No episode appearances found.</div>
                        ) : (
                            <div className="max-h-[70vh] overflow-auto divide-y divide-slate-100">
                                {appearances.map((appearance) => (
                                    <div key={appearance.video_id} className="p-4 flex items-start gap-3">
                                        <div className="w-24 h-14 rounded overflow-hidden bg-slate-100 border border-slate-200 shrink-0">
                                            {appearance.thumbnail_url ? (
                                                <img
                                                    src={toApiUrl(appearance.thumbnail_url)}
                                                    alt=""
                                                    className="w-full h-full object-cover"
                                                />
                                            ) : null}
                                        </div>
                                        <div className="min-w-0 flex-1 space-y-1">
                                            <Link
                                                to={`/video/${appearance.video_id}?t=${Math.floor(appearance.first_start_time)}`}
                                                className="font-medium text-slate-700 hover:text-blue-600 line-clamp-2"
                                            >
                                                {appearance.title}
                                            </Link>
                                            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                                                <span className="inline-flex items-center gap-1"><Calendar size={11} />{appearance.published_at ? new Date(appearance.published_at).toLocaleDateString() : 'No date'}</span>
                                                <span className="inline-flex items-center gap-1"><Hash size={11} />{appearance.segment_count} segments</span>
                                                <span>{formatDuration(appearance.total_speaking_time)} total</span>
                                            </div>
                                            <div className="flex flex-wrap items-center gap-2 text-xs">
                                                <Link
                                                    to={`/video/${appearance.video_id}?t=${Math.floor(appearance.first_start_time)}`}
                                                    className="text-blue-600 hover:text-blue-700 font-medium"
                                                >
                                                    Open at first appearance ({formatDuration(appearance.first_start_time)})
                                                </Link>
                                                <span className="text-slate-300">|</span>
                                                <Link
                                                    to={`/video/${appearance.video_id}`}
                                                    className="text-slate-500 hover:text-slate-700"
                                                >
                                                    Open episode
                                                </Link>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </section>

                <section className="space-y-3 min-w-0">
                    <div className="flex items-center gap-2 text-slate-700">
                        <AudioLines size={16} />
                        <h2 className="text-lg font-semibold">Voice Profiles</h2>
                    </div>
                    <div className="glass-panel rounded-xl border border-slate-200/60 overflow-hidden">
                        <div className="p-3 text-xs text-slate-500 border-b border-slate-100 bg-slate-50">
                            Voice profiles are embedding samples used for future speaker matching. Removing one does not rewrite existing transcript assignments.
                        </div>
                        {previewProfile && canPreviewProfile(previewProfile) && (
                            <div className="p-3 border-b border-slate-100 bg-white space-y-3">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                    <div className="text-sm font-medium text-slate-700">
                                        Previewing profile #{previewProfile.id}
                                        {previewProfile.sample_start_time != null && previewProfile.sample_end_time != null && (
                                            <span className="ml-2 text-xs font-normal text-slate-500">
                                                {formatDuration(previewProfile.sample_start_time)} - {formatDuration(previewProfile.sample_end_time)}
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        {previewProfile.source_video_id && previewProfile.sample_start_time != null && (
                                            <Link
                                                to={`/video/${previewProfile.source_video_id}?t=${Math.floor(previewProfile.sample_start_time)}`}
                                                className="text-xs text-blue-600 hover:text-blue-700 font-medium"
                                            >
                                                Open episode at clip
                                            </Link>
                                        )}
                                        <button
                                            type="button"
                                            onClick={closeProfilePreview}
                                            className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                                            title="Close preview"
                                        >
                                            <X size={12} />
                                            Close
                                        </button>
                                    </div>
                                </div>
                                <div className="rounded-lg overflow-hidden border border-slate-200 bg-black">
                                    <div className="aspect-video">
                                        {previewUsesYoutube ? (
                                            <YouTube
                                                key={`profile-preview-${previewProfile.id}`}
                                                videoId={previewProfile.source_video_youtube_id}
                                                className="w-full h-full"
                                                iframeClassName="w-full h-full"
                                                opts={{
                                                    width: '100%',
                                                    height: '100%',
                                                    playerVars: {
                                                        autoplay: 1,
                                                        controls: 1,
                                                        enablejsapi: 1,
                                                        rel: 0,
                                                        modestbranding: 1,
                                                        start: Math.max(0, Math.floor(previewProfile.sample_start_time || 0)),
                                                    },
                                                }}
                                                onReady={handleProfilePreviewReady}
                                            />
                                        ) : previewIsAudioOnly ? (
                                            <div className="flex h-full items-center justify-center p-6">
                                                <audio
                                                    key={`profile-preview-audio-${previewProfile.id}`}
                                                    controls
                                                    className="w-full"
                                                    src={previewMediaUrl}
                                                    onLoadedMetadata={handleNativeProfilePreviewLoaded}
                                                />
                                            </div>
                                        ) : (
                                            <video
                                                key={`profile-preview-local-${previewProfile.id}`}
                                                controls
                                                playsInline
                                                className="h-full w-full bg-black object-contain"
                                                src={previewMediaUrl}
                                                onLoadedMetadata={handleNativeProfilePreviewLoaded}
                                            />
                                        )}
                                    </div>
                                </div>
                                {previewProfile.sample_text && (
                                    <blockquote className="text-xs italic text-slate-600 bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                                        "{previewProfile.sample_text}"
                                    </blockquote>
                                )}
                            </div>
                        )}
                        {profiles.length === 0 ? (
                            <div className="p-6 text-center text-slate-400 text-sm">No voice profiles found.</div>
                        ) : (
                            <div className="max-h-[70vh] overflow-auto divide-y divide-slate-100">
                                {profiles.map((profile, idx) => (
                                    <div key={profile.id} className="p-4 flex items-start justify-between gap-3">
                                        <div className="min-w-0 flex-1 space-y-1">
                                            <div className="flex items-center gap-2 text-xs text-slate-500">
                                                <span className="px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200">Profile #{profile.id}</span>
                                                {idx === 0 && <span className="px-1.5 py-0.5 rounded bg-blue-50 border border-blue-200 text-blue-700">Newest</span>}
                                            </div>
                                            <div className="text-sm text-slate-700">
                                                {profile.source_video_id ? (
                                                    <Link to={`/video/${profile.source_video_id}`} className="hover:text-blue-600">
                                                        {profile.source_video_title || `Video #${profile.source_video_id}`}
                                                    </Link>
                                                ) : (
                                                    <span>Imported / unknown source</span>
                                                )}
                                            </div>
                                            <div className="text-xs text-slate-500 flex flex-wrap items-center gap-2">
                                                <span>Added {new Date(profile.created_at).toLocaleString()}</span>
                                                {profile.source_video_published_at && (
                                                    <span>Episode date {new Date(profile.source_video_published_at).toLocaleDateString()}</span>
                                                )}
                                            </div>
                                            {(profile.sample_start_time != null || profile.sample_text) && (
                                                <div className="text-xs text-slate-600 space-y-1">
                                                    {profile.sample_start_time != null && (
                                                        <div>
                                                            Sample clip: {formatDuration(profile.sample_start_time)}
                                                            {profile.sample_end_time != null ? ` - ${formatDuration(profile.sample_end_time)}` : ''}
                                                        </div>
                                                    )}
                                                    {profile.sample_text && (
                                                        <div className="line-clamp-2 italic text-slate-500">
                                                            "{profile.sample_text}"
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                            {!profile.sample_text && profile.sample_start_time == null && (
                                                <div className="text-xs text-slate-400">Legacy profile (no stored sample clip metadata)</div>
                                            )}
                                            <div className="flex flex-wrap items-center gap-2 pt-1">
                                                {canPreviewProfile(profile) && (
                                                    <button
                                                        type="button"
                                                        onClick={() => {
                                                            if (previewProfile?.id === profile.id) {
                                                                closeProfilePreview();
                                                                return;
                                                            }
                                                            stopPreviewPlayback();
                                                            setPreviewPlayer(null);
                                                            setPreviewProfile(profile);
                                                        }}
                                                        className={`px-2.5 py-1 text-xs rounded-lg border inline-flex items-center gap-1 ${
                                                            previewProfile?.id === profile.id
                                                                ? 'border-slate-300 text-slate-700 bg-slate-100 hover:bg-slate-200'
                                                                : 'border-blue-200 text-blue-700 bg-blue-50 hover:bg-blue-100'
                                                        }`}
                                                    >
                                                        <Play size={12} />
                                                        {previewProfile?.id === profile.id ? 'Close Preview' : 'Preview Clip'}
                                                    </button>
                                                )}
                                                {profile.source_video_id && profile.sample_start_time != null && (
                                                    <Link
                                                        to={`/video/${profile.source_video_id}?t=${Math.floor(profile.sample_start_time)}`}
                                                        className="px-2.5 py-1 text-xs rounded-lg border border-slate-200 text-slate-600 bg-white hover:bg-slate-50"
                                                    >
                                                        Open at Clip
                                                    </Link>
                                                )}
                                            </div>
                                        </div>
                                        <div className="flex flex-col gap-2 shrink-0">
                                            <button
                                                type="button"
                                                onClick={() => void handleBulkReassignProfileSegments(profile)}
                                                disabled={reassigningProfileId === profile.id}
                                                className="px-2.5 py-1.5 text-xs rounded-lg border border-indigo-200 text-indigo-700 bg-indigo-50 hover:bg-indigo-100 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-1"
                                                title="Reassign all segments linked to this profile"
                                            >
                                                {reassigningProfileId === profile.id ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                                                Reassign Segments
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => void openMoveProfileDialog(profile)}
                                                disabled={profiles.length <= 1}
                                                className="px-2.5 py-1.5 text-xs rounded-lg border border-blue-200 text-blue-700 bg-blue-50 hover:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-1"
                                                title={profiles.length <= 1 ? 'Cannot move the last voice profile' : 'Move this profile to another/new speaker'}
                                            >
                                                <GitMerge size={12} />
                                                Move...
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => void handleDeleteProfile(profile)}
                                                disabled={deletingProfileId === profile.id || profiles.length <= 1}
                                                className="px-2.5 py-1.5 text-xs rounded-lg border border-red-200 text-red-700 bg-red-50 hover:bg-red-100 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-1"
                                                title={profiles.length <= 1 ? 'Cannot remove the last voice profile' : 'Remove this voice profile'}
                                            >
                                                {deletingProfileId === profile.id ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                                                Remove
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </section>
            </div>

            {editingSpeaker && (
                <SpeakerModal
                    speaker={editingSpeaker}
                    onClose={() => setEditingSpeaker(null)}
                    onUpdate={(updatedSpeaker) => {
                        setEditingSpeaker(updatedSpeaker);
                        setSpeaker(prev => prev ? {
                            ...prev,
                            ...updatedSpeaker,
                            total_speaking_time: prev.total_speaking_time,
                            embedding_count: prev.embedding_count,
                        } : updatedSpeaker);
                    }}
                    onMerge={(mergedIntoSpeaker) => {
                        setEditingSpeaker(null);
                        if (mergedIntoSpeaker?.id) {
                            navigate(`/speakers/${mergedIntoSpeaker.id}`);
                            return;
                        }
                        void fetchData();
                    }}
                />
            )}

            {movingProfile && speaker && (
                <div className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4" onClick={() => !movingProfileSubmit && setMovingProfile(null)}>
                    <div className="w-full max-w-2xl bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden" onClick={(e) => e.stopPropagation()}>
                        <div className="p-4 border-b border-slate-100 flex items-center justify-between gap-3">
                            <div>
                                <h3 className="text-lg font-semibold text-slate-800">Move Voice Profile #{movingProfile.id}</h3>
                                <p className="text-xs text-slate-500 mt-0.5">
                                    Move this profile to an existing speaker, or create a new speaker from it.
                                </p>
                            </div>
                            <button
                                type="button"
                                onClick={() => setMovingProfile(null)}
                                disabled={movingProfileSubmit}
                                className="p-2 rounded-full hover:bg-slate-100 text-slate-500 disabled:opacity-50"
                            >
                                <X size={16} />
                            </button>
                        </div>

                        <div className="p-4 space-y-4 max-h-[75vh] overflow-y-auto">
                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600 space-y-1">
                                <div><span className="font-medium text-slate-700">Source speaker:</span> {speaker.name}</div>
                                {movingProfile.sample_start_time != null && (
                                    <div><span className="font-medium text-slate-700">Sample clip:</span> {formatDuration(movingProfile.sample_start_time)}{movingProfile.sample_end_time != null ? ` - ${formatDuration(movingProfile.sample_end_time)}` : ''}</div>
                                )}
                                {movingProfile.sample_text && <div className="italic line-clamp-2">"{movingProfile.sample_text}"</div>}
                            </div>

                            <div className="rounded-xl border border-slate-200 p-3 space-y-3">
                                <div className="flex items-center gap-2 text-sm font-medium text-slate-700">
                                    <GitMerge size={14} className="text-blue-600" />
                                    Move To Existing Speaker
                                </div>
                                <div className="relative">
                                    <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400" />
                                    <input
                                        type="text"
                                        value={moveSearch}
                                        onChange={(e) => setMoveSearch(e.target.value)}
                                        placeholder="Search speakers..."
                                        className="w-full pl-8 pr-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400"
                                    />
                                </div>
                                <div className="max-h-56 overflow-y-auto rounded-lg border border-slate-200 divide-y divide-slate-100">
                                    {loadingMoveTargets ? (
                                        <div className="flex items-center justify-center p-4 text-slate-400 text-sm">
                                            <Loader2 size={16} className="animate-spin mr-2" /> Loading speakers...
                                        </div>
                                    ) : filteredMoveTargets.length === 0 ? (
                                        <div className="p-4 text-center text-sm text-slate-400">
                                            {moveSearch ? 'No matching speakers' : 'No other speakers available'}
                                        </div>
                                    ) : (
                                        filteredMoveTargets.map(target => (
                                            <button
                                                key={target.id}
                                                type="button"
                                                onClick={() => void handleMoveProfileToExisting(target)}
                                                disabled={movingProfileSubmit}
                                                className="w-full px-3 py-2.5 text-left hover:bg-blue-50 disabled:opacity-50 transition-colors flex items-center justify-between gap-2"
                                            >
                                                <div className="min-w-0">
                                                    <div className="text-sm font-medium text-slate-700 truncate">{target.name}</div>
                                                    <div className="text-xs text-slate-500">
                                                        {formatDuration(target.total_speaking_time)} total • {target.embedding_count ?? 0} profiles
                                                    </div>
                                                </div>
                                                <GitMerge size={14} className="text-blue-500 shrink-0" />
                                            </button>
                                        ))
                                    )}
                                </div>
                            </div>

                            <div className="rounded-xl border border-slate-200 p-3 space-y-3">
                                <div className="flex items-center gap-2 text-sm font-medium text-slate-700">
                                    <PlusCircle size={14} className="text-green-600" />
                                    Create New Speaker From This Profile
                                </div>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={newMoveSpeakerName}
                                        onChange={(e) => setNewMoveSpeakerName(e.target.value)}
                                        placeholder="New speaker name"
                                        className="flex-1 px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500/20 focus:border-green-400"
                                    />
                                    <button
                                        type="button"
                                        onClick={() => void handleMoveProfileToNewSpeaker()}
                                        disabled={movingProfileSubmit || !newMoveSpeakerName.trim()}
                                        className="px-3 py-2 text-sm rounded-lg border border-green-200 bg-green-50 text-green-700 hover:bg-green-100 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-1.5"
                                    >
                                        {movingProfileSubmit ? <Loader2 size={14} className="animate-spin" /> : <PlusCircle size={14} />}
                                        Create + Move
                                    </button>
                                </div>
                                <p className="text-xs text-slate-500">
                                    This creates a new speaker in the same channel and transfers this one voice profile to it.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
