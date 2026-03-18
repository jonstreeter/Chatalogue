import { useState, useRef, useEffect, memo } from 'react';
import YouTube from 'react-youtube';
import api from '../lib/api';
import { toApiUrl } from '../lib/api';
import type { Speaker } from '../types';
import { X, Check, Camera, Loader2, ChevronLeft, ChevronRight, UserMinus, UserPlus, GitMerge, Search, Trash2 } from 'lucide-react';

const SAMPLE_STOP_EARLY_SECONDS = 0.35;

interface SpeakerModalProps {
    speaker: Speaker;
    initialSample?: any; // Optional initial sample to play
    onClose: () => void;
    onUpdate: (updatedSpeaker: Speaker) => void;
    onMerge?: (mergedIntoSpeaker?: Speaker) => void; // Called after a successful merge to refresh parent data
}

function SpeakerModalComponent({ speaker, initialSample, onClose, onUpdate, onMerge }: SpeakerModalProps) {
    const [name, setName] = useState(speaker.name);
    const [savingName, setSavingName] = useState(false);
    const [nameError, setNameError] = useState<string | null>(null);
    const nameInputRef = useRef<HTMLInputElement>(null);
    const autoSelectNameRef = useRef(true);

    // Multiple samples state
    const [samples, setSamples] = useState<any[]>(initialSample ? [initialSample] : []);
    const [currentSampleIndex, setCurrentSampleIndex] = useState(0);
    const [loading, setLoading] = useState(!initialSample);
    const [player, setPlayer] = useState<any>(null);
    const [isCropping, setIsCropping] = useState(false);
    const [extracting, setExtracting] = useState(false);
    const [deletingThumbnail, setDeletingThumbnail] = useState(false);

    // Merge state
    const [showMerge, setShowMerge] = useState(false);
    const [channelSpeakers, setChannelSpeakers] = useState<Speaker[]>([]);
    const [mergeSearch, setMergeSearch] = useState('');
    const [merging, setMerging] = useState(false);
    const [loadingSpeakers, setLoadingSpeakers] = useState(false);

    // Crop state
    const [cropStart, setCropStart] = useState<{ x: number, y: number } | null>(null);
    const [cropRect, setCropRect] = useState<{ x: number, y: number, w: number, h: number } | null>(null);
    const overlayRef = useRef<HTMLDivElement>(null);

    // Derived from current sample
    const currentSample = samples[currentSampleIndex];
    const videoId = currentSample?.youtube_id;
    const backendVideoId = currentSample?.video_id;
    const startTime = currentSample?.start_time || 0;
    const sampleText = currentSample?.text;
    const isGeneratedSpeakerName = (value?: string | null) => /^Speaker\s+\d+$/i.test((value || '').trim());

    // Fetch multiple samples on mount (top 5 longest)
    useEffect(() => {
        const fetchSamples = async () => {
            try {
                const res = await api.get<any[]>(`/speakers/${speaker.id}/samples?strategy=longest&count=5`);
                if (res.data && res.data.length > 0) {
                    // If we have an initial sample, append new ones that aren't duplicates
                    if (initialSample) {
                        const newSamples = res.data.filter(s =>
                            !(s.video_id === initialSample.video_id && Math.abs(s.start_time - initialSample.start_time) < 1)
                        );
                        setSamples(prev => [...prev, ...newSamples]);
                    } else {
                        setSamples(res.data);
                    }
                }
            } catch (e) {
                console.error("Failed to fetch samples", e);
            } finally {
                setLoading(false);
            }
        };
        fetchSamples();
    }, [speaker.id, initialSample]);

    useEffect(() => {
        setName(speaker.name);
        autoSelectNameRef.current = isGeneratedSpeakerName(speaker.name);
    }, [speaker.id, speaker.name]);

    // Fetch channel speakers for merge dropdown
    const fetchChannelSpeakers = async () => {
        setLoadingSpeakers(true);
        try {
            const res = await api.get<Speaker[]>('/speakers', { params: { channel_id: speaker.channel_id } });
            setChannelSpeakers(res.data.filter(s => s.id !== speaker.id));
        } catch (e) {
            console.error('Failed to fetch speakers', e);
        } finally {
            setLoadingSpeakers(false);
        }
    };

    const handleOpenMerge = () => {
        setShowMerge(true);
        fetchChannelSpeakers();
    };

    const handleMergeInto = async (targetSpeaker: Speaker) => {
        if (!confirm(`Merge "${speaker.name}" into "${targetSpeaker.name}"?\n\nAll segments and voice samples will be reassigned to "${targetSpeaker.name}".`)) return;
        setMerging(true);
        try {
            const res = await api.post<Speaker>('/speakers/merge', {
                target_id: targetSpeaker.id,
                source_ids: [speaker.id]
            });
            onMerge?.(res.data);
            onClose();
        } catch (e) {
            console.error('Failed to merge', e);
            alert('Failed to merge speakers');
        } finally {
            setMerging(false);
        }
    };

    const filteredMergeSpeakers = channelSpeakers.filter(s =>
        s.name.toLowerCase().includes(mergeSearch.toLowerCase())
    );

    // Navigate to previous/next sample
    const goToPrevSample = () => {
        if (currentSampleIndex > 0) {
            const newIndex = currentSampleIndex - 1;
            const nextSample = samples[newIndex];
            const sameVideo = nextSample && currentSample && nextSample.youtube_id === currentSample.youtube_id;
            setCurrentSampleIndex(newIndex);
            setCropRect(null);
            setIsCropping(false);
            // If it's the same episode, seek the existing player immediately.
            // If it's a different episode, let the remounted player handle start time in onReady.
            if (sameVideo && player && nextSample) {
                try {
                    player.seekTo(nextSample.start_time, true);
                    player.playVideo?.();
                } catch (e) {
                    console.warn('Failed to seek previous speaker sample', e);
                }
            }
        }
    };

    const goToNextSample = () => {
        if (currentSampleIndex < samples.length - 1) {
            const newIndex = currentSampleIndex + 1;
            const nextSample = samples[newIndex];
            const sameVideo = nextSample && currentSample && nextSample.youtube_id === currentSample.youtube_id;
            setCurrentSampleIndex(newIndex);
            setCropRect(null);
            setIsCropping(false);
            // If it's the same episode, seek the existing player immediately.
            // If it's a different episode, let the remounted player handle start time in onReady.
            if (sameVideo && player && nextSample) {
                try {
                    player.seekTo(nextSample.start_time, true);
                    player.playVideo?.();
                } catch (e) {
                    console.warn('Failed to seek next speaker sample', e);
                }
            }
        }
    };

    const handleSaveName = async () => {
        const trimmed = name.trim();
        const currentName = (speaker.name || '').trim();
        if (!trimmed) {
            setNameError('Name cannot be empty.');
            return;
        }
        if (trimmed === currentName) {
            if (name !== trimmed) {
                setName(trimmed);
            }
            setNameError(null);
            return;
        }
        setSavingName(true);
        setNameError(null);
        try {
            const res = await api.patch(`/speakers/${speaker.id}`, { name: trimmed });
            setName(trimmed);
            onUpdate(res.data);
        } catch (e: any) {
            console.error("Failed to update name", e);
            const detail = e?.response?.data?.detail || 'Failed to update name';
            setNameError(detail);
            alert(detail);
        } finally {
            setSavingName(false);
        }
    };

    const handleToggleExtra = async () => {
        try {
            const res = await api.patch(`/speakers/${speaker.id}`, { is_extra: !speaker.is_extra });
            onUpdate(res.data);
            onClose(); // Close modal after moving to extras
        } catch (e) {
            console.error("Failed to toggle extra status", e);
            alert("Failed to update speaker");
        }
    };

    // Player Ready
    const onReady = (event: any) => {
        setPlayer(event.target);
        event.target.seekTo(startTime);
        event.target.playVideo();
    };

    // Keep playback scoped to the selected sample clip to avoid overlapping audio
    // with the main page and to prevent the modal preview from running past the quote.
    useEffect(() => {
        if (!player || !currentSample?.end_time) return;

        const sampleStart = currentSample.start_time || 0;
        const sampleEnd = currentSample.end_time;

        const interval = setInterval(() => {
            try {
                if (typeof player.getCurrentTime !== 'function' || typeof player.pauseVideo !== 'function') return;
                const t = player.getCurrentTime();
                const stopAt = Math.max(sampleStart, sampleEnd - SAMPLE_STOP_EARLY_SECONDS);
                // Only auto-pause if playback has actually reached this sample window.
                if (t >= sampleStart && t >= stopAt) {
                    player.pauseVideo();
                }
            } catch {
                // Ignore transient player API errors while iframe/player reinitializes.
            }
        }, 200);

        return () => clearInterval(interval);
    }, [player, currentSampleIndex, currentSample?.start_time, currentSample?.end_time]);

    // Crop Interaction - track whether we're drawing a new box or dragging an existing one
    const [isDragging, setIsDragging] = useState(false);
    const [dragOffset, setDragOffset] = useState<{ x: number, y: number } | null>(null);

    const isInsideCropRect = (mouseX: number, mouseY: number): boolean => {
        if (!cropRect || cropRect.w === 0) return false;
        return mouseX >= cropRect.x && mouseX <= cropRect.x + cropRect.w &&
            mouseY >= cropRect.y && mouseY <= cropRect.y + cropRect.h;
    };

    const handleMouseDown = (e: React.MouseEvent) => {
        if (!isCropping || !overlayRef.current) return;

        // Ignore clicks on controls (e.g. Save/Cancel buttons) to prevent redrawing
        if ((e.target as HTMLElement).closest('button')) {
            return;
        }

        const rect = overlayRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // If clicking inside existing crop box, start dragging it
        if (cropRect && cropRect.w > 0 && isInsideCropRect(x, y)) {
            setIsDragging(true);
            setDragOffset({ x: x - cropRect.x, y: y - cropRect.y });
        } else {
            // Start new crop
            setCropStart({ x, y });
            setCropRect({ x, y, w: 0, h: 0 });
            setIsDragging(false);
        }
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!isCropping || !overlayRef.current) return;
        const rect = overlayRef.current.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;

        // If dragging existing box, move it
        if (isDragging && cropRect && dragOffset) {
            const newX = Math.max(0, Math.min(currentX - dragOffset.x, rect.width - cropRect.w));
            const newY = Math.max(0, Math.min(currentY - dragOffset.y, rect.height - cropRect.h));
            setCropRect({ ...cropRect, x: newX, y: newY });
            return;
        }

        // If drawing new box
        if (!cropStart) return;

        // Calculate the size based on the larger dimension, ensuring 1:1 ratio
        const deltaX = currentX - cropStart.x;
        const deltaY = currentY - cropStart.y;
        const size = Math.max(Math.abs(deltaX), Math.abs(deltaY));

        // Adjust position based on drag direction
        const x = deltaX < 0 ? cropStart.x - size : cropStart.x;
        const y = deltaY < 0 ? cropStart.y - size : cropStart.y;
        // Clamp square selection inside the overlay bounds so backend crop coords
        // never exceed 0..1, which would make ffmpeg crop fail.
        const clampedX = Math.max(0, Math.min(x, rect.width));
        const clampedY = Math.max(0, Math.min(y, rect.height));
        const maxSizeByBounds = Math.max(0, Math.min(
            size,
            rect.width - clampedX,
            rect.height - clampedY
        ));

        setCropRect({ x: clampedX, y: clampedY, w: maxSizeByBounds, h: maxSizeByBounds });
    };

    const handleMouseUp = () => {
        setCropStart(null);
        setIsDragging(false);
        setDragOffset(null);
    };

    const handleExtractThumbnail = async () => {
        if (!player || !cropRect || !overlayRef.current || !backendVideoId) return;

        setExtracting(true);
        try {
            const timestamp = player.getCurrentTime();

            // Convert to relative 0-1 coords
            const containerW = overlayRef.current.clientWidth;
            const containerH = overlayRef.current.clientHeight;

            const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
            const minPx = 12;
            const safeRect = {
                x: Math.max(0, Math.min(cropRect.x, containerW - minPx)),
                y: Math.max(0, Math.min(cropRect.y, containerH - minPx)),
                w: Math.max(minPx, Math.min(cropRect.w, containerW)),
                h: Math.max(minPx, Math.min(cropRect.h, containerH))
            };
            // Ensure width/height do not overflow the overlay bounds
            safeRect.w = Math.min(safeRect.w, containerW - safeRect.x);
            safeRect.h = Math.min(safeRect.h, containerH - safeRect.y);

            const relCoords = {
                x: clamp01(safeRect.x / containerW),
                y: clamp01(safeRect.y / containerH),
                w: clamp01(safeRect.w / containerW),
                h: clamp01(safeRect.h / containerH)
            };

            const res = await api.post(`/speakers/${speaker.id}/thumbnail/extract`, {
                video_id: backendVideoId,
                timestamp: timestamp,
                crop_coords: relCoords
            });

            onUpdate(res.data);
            setIsCropping(false);
            setCropRect(null);

        } catch (e: any) {
            console.error("Failed to extract thumbnail", e);
            alert(e?.response?.data?.detail || "Failed to extract thumbnail");
        } finally {
            setExtracting(false);
        }
    };

    const handleDeleteThumbnail = async () => {
        if (!speaker.thumbnail_path) return;
        if (!confirm(`Delete thumbnail for "${speaker.name}"?`)) return;

        setDeletingThumbnail(true);
        try {
            const res = await api.delete(`/speakers/${speaker.id}/thumbnail`);
            onUpdate(res.data);
            setIsCropping(false);
            setCropRect(null);
        } catch (e: any) {
            console.error("Failed to delete thumbnail", e);
            alert(e?.response?.data?.detail || "Failed to delete thumbnail");
        } finally {
            setDeletingThumbnail(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4" onClick={onClose}>
            <div className="bg-white rounded-2xl w-full max-w-6xl h-[85vh] flex overflow-hidden shadow-2xl" onClick={e => e.stopPropagation()}>

                {/* Left: Video */}
                <div className="w-2/3 bg-black relative flex items-center justify-center">
                    {loading ? (
                        <div className="text-white flex items-center gap-2">
                            <Loader2 className="animate-spin" /> Loading Video...
                        </div>
                    ) : videoId ? (
                        <div className="relative w-full aspect-video group shadow-lg mx-auto">
                            <YouTube
                                key={videoId || `sample-${currentSampleIndex}`}
                                videoId={videoId}
                                className="w-full h-full"
                                iframeClassName="w-full h-full"
                                opts={{
                                    height: '100%',
                                    width: '100%',
                                    playerVars: {
                                        autoplay: 1,
                                        controls: 1,
                                        enablejsapi: 1,
                                        rel: 0,
                                        modestbranding: 1
                                    },
                                }}
                                onReady={onReady}
                            />

                            {/* Crop Overlay */}
                            {isCropping && (
                                <div
                                    ref={overlayRef}
                                    className="absolute inset-0 z-10 cursor-crosshair"
                                    onMouseDown={handleMouseDown}
                                    onMouseMove={handleMouseMove}
                                    onMouseUp={handleMouseUp}
                                    onMouseLeave={handleMouseUp}
                                >
                                    {/* Dim overlay when cropping */}
                                    {!cropRect && (
                                        <div className="absolute inset-0 bg-black/30 flex items-center justify-center">
                                            <div className="text-white text-center">
                                                <p className="text-lg font-medium">Draw a square around the face</p>
                                                <p className="text-sm opacity-75 mt-1">Click and drag to select</p>
                                            </div>
                                        </div>
                                    )}

                                    {cropRect && cropRect.w > 0 && (
                                        <>
                                            <div
                                                className="absolute border-4 border-green-400 rounded-lg shadow-[0_0_0_9999px_rgba(0,0,0,0.6)] cursor-move"
                                                style={{
                                                    left: cropRect.x,
                                                    top: cropRect.y,
                                                    width: cropRect.w,
                                                    height: cropRect.h
                                                }}
                                            />
                                            {/* Save/Cancel buttons on the video */}
                                            <div
                                                className="absolute flex gap-2 z-20"
                                                style={{
                                                    left: cropRect.x,
                                                    top: cropRect.y + cropRect.h + 8
                                                }}
                                                onMouseDown={(e) => e.stopPropagation()}
                                            >
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); handleExtractThumbnail(); }}
                                                    onMouseDown={(e) => e.stopPropagation()}
                                                    disabled={extracting}
                                                    className="px-4 py-2 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 disabled:opacity-50 flex items-center gap-2 shadow-lg"
                                                >
                                                    {extracting ? <Loader2 className="animate-spin" size={16} /> : <Check size={16} />}
                                                    Save
                                                </button>
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); setIsCropping(false); setCropRect(null); }}
                                                    onMouseDown={(e) => e.stopPropagation()}
                                                    className="px-4 py-2 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 flex items-center gap-2 shadow-lg"
                                                >
                                                    <X size={16} /> Cancel
                                                </button>
                                            </div>
                                        </>
                                    )}
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-white text-lg">No video sample available for this speaker</div>
                    )}
                </div>

                {/* Right: Controls */}
                <div className="w-1/3 flex flex-col border-l border-slate-100 bg-slate-50">
                    <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-white">
                        <h2 className="text-xl font-bold text-slate-800">Edit Speaker</h2>
                        <button onClick={onClose} className="p-2 hover:bg-slate-100 rounded-full transition-colors">
                            <X size={20} className="text-slate-500" />
                        </button>
                    </div>

                    <div className="p-6 space-y-8 flex-1 overflow-y-auto">
                        {/* Transcript Quote with Sample Navigation */}
                        {samples.length > 0 && (
                            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-5 rounded-xl border border-blue-100 shadow-sm">
                                <div className="flex justify-between items-center mb-3">
                                    <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide">This speaker said:</p>
                                    {samples.length > 1 && (
                                        <div className="flex items-center gap-1">
                                            <button
                                                onClick={goToPrevSample}
                                                disabled={currentSampleIndex === 0}
                                                className="p-1 rounded hover:bg-blue-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                                            >
                                                <ChevronLeft size={16} className="text-blue-600" />
                                            </button>
                                            <span className="text-xs font-medium text-blue-600 min-w-[3rem] text-center">
                                                {currentSampleIndex + 1} / {samples.length}
                                            </span>
                                            <button
                                                onClick={goToNextSample}
                                                disabled={currentSampleIndex === samples.length - 1}
                                                className="p-1 rounded hover:bg-blue-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                                            >
                                                <ChevronRight size={16} className="text-blue-600" />
                                            </button>
                                        </div>
                                    )}
                                </div>
                                <blockquote className="text-slate-700 italic leading-relaxed">
                                    "{sampleText}"
                                </blockquote>
                                <p className="text-xs text-slate-400 mt-2">
                                    Duration: {currentSample ? ((currentSample.end_time - currentSample.start_time).toFixed(1)) : 0}s
                                </p>
                            </div>
                        )}

                        {/* Name Section */}
                        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
                            <label className="block text-sm font-semibold text-slate-600 mb-3">Speaker Name</label>
                            <div className="flex gap-2">
                                <input
                                    ref={nameInputRef}
                                    value={name}
                                    onChange={e => {
                                        setName(e.target.value);
                                        autoSelectNameRef.current = false;
                                        if (nameError) setNameError(null);
                                    }}
                                    onFocus={(e) => {
                                        if (autoSelectNameRef.current && isGeneratedSpeakerName(name)) {
                                            e.currentTarget.select();
                                        }
                                    }}
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        if (autoSelectNameRef.current && isGeneratedSpeakerName(name)) {
                                            e.currentTarget.select();
                                        }
                                    }}
                                    onKeyDown={(e) => {
                                        e.stopPropagation();
                                        if (e.key === 'Enter') {
                                            e.preventDefault();
                                            if (!savingName) {
                                                void handleSaveName();
                                            }
                                        }
                                    }}
                                    onMouseDown={(e) => e.stopPropagation()}
                                    className="flex-1 px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all placeholder:text-slate-400 font-medium"
                                    placeholder="Enter name..."
                                />
                                <button
                                    onClick={handleSaveName}
                                    disabled={savingName || name.trim() === (speaker.name || '').trim()}
                                    className="p-2.5 bg-blue-600 text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors shadow-sm"
                                    title="Save Name"
                                >
                                    {savingName ? <Loader2 size={20} className="animate-spin" /> : <Check size={20} />}
                                </button>
                            </div>
                            {nameError && (
                                <p className="mt-2 text-xs text-red-600">{nameError}</p>
                            )}
                        </div>

                        {/* Merge with Existing Speaker */}
                        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-3">
                            <div className="flex justify-between items-center">
                                <div>
                                    <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                                        <GitMerge size={16} className="text-purple-500" />
                                        Merge into Another Speaker
                                    </h3>
                                    <p className="text-xs text-slate-500 mt-0.5">Reassign all segments to an existing speaker</p>
                                </div>
                                {!showMerge && (
                                    <button
                                        onClick={handleOpenMerge}
                                        className="px-3 py-1.5 bg-purple-50 text-purple-600 rounded-lg text-sm font-medium hover:bg-purple-100 transition-colors"
                                    >
                                        Choose Speaker
                                    </button>
                                )}
                            </div>
                            {showMerge && (
                                <div className="space-y-2">
                                    <div className="relative">
                                        <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400" />
                                        <input
                                            type="text"
                                            value={mergeSearch}
                                            onChange={(e) => setMergeSearch(e.target.value)}
                                            placeholder="Search speakers..."
                                            className="w-full pl-8 pr-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-400"
                                            autoFocus
                                        />
                                    </div>
                                    <div className="max-h-48 overflow-y-auto rounded-lg border border-slate-200 divide-y divide-slate-100">
                                        {loadingSpeakers ? (
                                            <div className="flex items-center justify-center p-4 text-slate-400">
                                                <Loader2 size={16} className="animate-spin mr-2" /> Loading...
                                            </div>
                                        ) : filteredMergeSpeakers.length === 0 ? (
                                            <div className="p-4 text-center text-sm text-slate-400">
                                                {mergeSearch ? 'No matching speakers' : 'No other speakers available'}
                                            </div>
                                        ) : (
                                            filteredMergeSpeakers.map(s => (
                                                <button
                                                    key={s.id}
                                                    onClick={() => handleMergeInto(s)}
                                                    disabled={merging}
                                                    className="w-full flex items-center gap-3 px-3 py-2.5 hover:bg-purple-50 transition-colors text-left disabled:opacity-50"
                                                >
                                                    {s.thumbnail_path ? (
                                                        <img
                                                            src={toApiUrl(s.thumbnail_path)}
                                                            className="w-8 h-8 rounded-full object-cover border border-slate-200"
                                                        />
                                                    ) : (
                                                        <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 text-xs font-medium border border-slate-200">
                                                            {s.name.charAt(0).toUpperCase()}
                                                        </div>
                                                    )}
                                                    <div className="flex-1 min-w-0">
                                                        <span className="text-sm font-medium text-slate-700 block truncate">{s.name}</span>
                                                        <span className="text-[10px] text-slate-400">{s.total_speaking_time?.toFixed(0)}s total</span>
                                                    </div>
                                                    <GitMerge size={14} className="text-purple-400 shrink-0" />
                                                </button>
                                            ))
                                        )}
                                    </div>
                                    <button
                                        onClick={() => { setShowMerge(false); setMergeSearch(''); }}
                                        className="w-full py-1.5 text-xs text-slate-400 hover:text-slate-600 transition-colors"
                                    >
                                        Cancel
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Mark as Extra Button */}
                        <button
                            onClick={handleToggleExtra}
                            className={`w-full py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all ${speaker.is_extra
                                ? 'bg-green-50 text-green-700 border border-green-200 hover:bg-green-100'
                                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                                }`}
                        >
                            {speaker.is_extra ? (
                                <><UserPlus size={18} /> Restore to Main Speakers</>
                            ) : (
                                <><UserMinus size={18} /> Move to Extras</>
                            )}
                        </button>

                        {/* Thumbnail Section */}
                        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
                            <div>
                                <h3 className="font-semibold text-slate-800">Thumbnail</h3>
                                <p className="text-sm text-slate-500 mt-1">Capture a face from the video</p>
                            </div>

                            <div className="flex items-center justify-end gap-2">
                                {speaker.thumbnail_path && (
                                    <button
                                        type="button"
                                        onClick={handleDeleteThumbnail}
                                        disabled={deletingThumbnail}
                                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-red-200 bg-red-50 text-red-700 hover:bg-red-100 disabled:opacity-50 disabled:cursor-not-allowed"
                                        title="Delete thumbnail"
                                    >
                                        {deletingThumbnail ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                                        Delete Thumbnail
                                    </button>
                                )}
                            </div>

                            <div className="flex justify-center pt-2 relative">
                                {/* Camera Button */}
                                <button
                                    onClick={() => {
                                        setIsCropping(!isCropping);
                                        if (!isCropping && player) {
                                            player.pauseVideo();
                                        }
                                        setCropRect(null);
                                    }}
                                    className={`absolute top-0 right-0 p-2 rounded-full shadow-md transition-all z-10 ${isCropping
                                        ? 'bg-red-50 text-red-600 border border-red-200 hover:bg-red-100'
                                        : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'
                                        }`}
                                    title={isCropping ? "Cancel Crop" : "Capture New Thumbnail"}
                                >
                                    {isCropping ? <X size={16} /> : <Camera size={16} />}
                                </button>

                                {speaker.thumbnail_path ? (
                                    <div className="relative group">
                                        <img
                                            src={toApiUrl(speaker.thumbnail_path)}
                                            className="w-40 h-40 rounded-full object-cover border-4 border-slate-100 shadow-lg"
                                            key={speaker.thumbnail_path}
                                        />
                                        <div className="absolute inset-0 rounded-full border border-black/5 pointer-events-none"></div>
                                    </div>
                                ) : (
                                    <div className="w-40 h-40 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 border-4 border-slate-50 shadow-inner">
                                        <span className="text-sm font-medium">No Image</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div >
    );
}

function areSpeakerModalPropsEqual(prev: SpeakerModalProps, next: SpeakerModalProps): boolean {
    const prevSpeaker = prev.speaker;
    const nextSpeaker = next.speaker;
    const sameSpeaker =
        prevSpeaker.id === nextSpeaker.id &&
        prevSpeaker.name === nextSpeaker.name &&
        prevSpeaker.thumbnail_path === nextSpeaker.thumbnail_path &&
        prevSpeaker.is_extra === nextSpeaker.is_extra &&
        prevSpeaker.embedding_count === nextSpeaker.embedding_count &&
        prevSpeaker.total_speaking_time === nextSpeaker.total_speaking_time;

    const prevSample = prev.initialSample;
    const nextSample = next.initialSample;
    const sameInitialSample =
        (!prevSample && !nextSample) ||
        (!!prevSample &&
            !!nextSample &&
            prevSample.video_id === nextSample.video_id &&
            prevSample.youtube_id === nextSample.youtube_id &&
            prevSample.start_time === nextSample.start_time &&
            prevSample.end_time === nextSample.end_time);

    // Ignore callback identity changes from parent polling updates.
    return sameSpeaker && sameInitialSample;
}

export const SpeakerModal = memo(SpeakerModalComponent, areSpeakerModalPropsEqual);
