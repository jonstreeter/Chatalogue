import { CheckCircle2, Loader2, Scissors, X } from 'lucide-react';
import { useClipsStore } from '../../../store/useClipsStore';

type Props = {
    videoId: number;
    onClipCreated: () => void;
};

export function ClipCreationPanel({ videoId, onClipCreated }: Props) {
    const clipSelection = useClipsStore((s) => s.clipSelection);
    const clipTitle = useClipsStore((s) => s.clipTitle);
    const creatingClip = useClipsStore((s) => s.creatingClip);
    const setClipTitle = useClipsStore((s) => s.setClipTitle);
    const setClipSelection = useClipsStore((s) => s.setClipSelection);
    const createClipFromSelection = useClipsStore((s) => s.createClipFromSelection);

    if (!clipSelection) return null;

    const handleCreateClip = async () => {
        const createdClip = await createClipFromSelection(videoId);
        if (createdClip) onClipCreated();
    };

    return (
        <div className="absolute bottom-0 left-0 right-0 bg-white border-t border-slate-200 p-4 shadow-lg animate-in slide-in-from-bottom-10 z-20">
            <div className="flex justify-between items-start mb-3">
                <div>
                    <h3 className="text-sm font-semibold text-slate-800 flex items-center gap-2">
                        <Scissors size={14} className="text-purple-500" />
                        Create Clip
                    </h3>
                    <p className="text-xs text-slate-500 font-mono mt-1">
                        {new Date(clipSelection.start * 1000).toISOString().substr(14, 5)} - {new Date(clipSelection.end * 1000).toISOString().substr(14, 5)}
                        <span className="mx-2">•</span>
                        {(clipSelection.end - clipSelection.start).toFixed(1)}s
                    </p>
                </div>
                <button
                    onClick={() => setClipSelection(null)}
                    className="text-slate-400 hover:text-slate-600"
                >
                    <X size={16} />
                </button>
            </div>

            <div className="space-y-3">
                <input
                    type="text"
                    value={clipTitle}
                    onChange={(e) => setClipTitle(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500"
                    placeholder="Clip Title..."
                    autoFocus
                />
                <button
                    onClick={handleCreateClip}
                    disabled={creatingClip || !clipTitle.trim()}
                    className="w-full flex items-center justify-center gap-2 bg-purple-600 text-white py-2 rounded-lg text-sm font-medium hover:bg-purple-700 transition-colors disabled:opacity-50"
                >
                    {creatingClip ? <Loader2 size={16} className="animate-spin" /> : <CheckCircle2 size={16} />}
                    Save Clip
                </button>
            </div>
        </div>
    );
}
