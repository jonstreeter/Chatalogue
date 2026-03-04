import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { Loader2, RefreshCw } from 'lucide-react';
import api from '../../lib/api';
import { SpeakerList } from '../../components/SpeakerList';

export function ChannelSpeakers() {
    const { id } = useParams<{ id: string }>();
    const [runningBulkRedo, setRunningBulkRedo] = useState(false);

    const handleBulkRedoDiarization = async () => {
        if (!id) return;
        setRunningBulkRedo(true);
        try {
            const dryRun = await api.post(`/channels/${id}/redo-diarization`, null, {
                params: { dry_run: true, processed_only: true, include_muted: false },
            });
            const counts = dryRun.data?.counts || {};
            const eligible = Number(counts.eligible || 0);
            if (eligible <= 0) {
                alert('No eligible processed episodes with raw transcripts were found for channel-wide re-diarization.');
                return;
            }

            const ok = confirm(
                `Queue re-diarization for ${eligible} episode(s)?\n\n` +
                `Scanned: ${counts.scanned || 0}\n` +
                `Skipped active jobs: ${counts.skipped_active || 0}\n` +
                `Skipped no raw transcript: ${counts.skipped_no_raw_transcript || 0}\n` +
                `Skipped muted: ${counts.skipped_muted || 0}`
            );
            if (!ok) return;

            const run = await api.post(`/channels/${id}/redo-diarization`, null, {
                params: { dry_run: false, processed_only: true, include_muted: false },
            });
            const rc = run.data?.counts || {};
            alert(
                `Queued ${rc.queued || 0} re-diarization job(s).\n\n` +
                `Deleted transcript segments: ${rc.deleted_segments || 0}\n` +
                `Deleted funny moments: ${rc.deleted_funny_moments || 0}\n` +
                `Errors: ${rc.errors || 0}`
            );
        } catch (e: any) {
            console.error('Bulk re-diarization failed', e);
            alert(e?.response?.data?.detail || 'Failed to run bulk re-diarization');
        } finally {
            setRunningBulkRedo(false);
        }
    };

    return (
        <div className="space-y-3">
            <div className="glass-panel rounded-xl border border-slate-200/70 p-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                <div className="text-xs text-slate-600">
                    <div className="font-medium text-slate-700">Legacy Cleanup</div>
                    <div>Bulk re-diarize processed episodes in this channel using existing raw transcripts.</div>
                </div>
                <button
                    type="button"
                    onClick={() => void handleBulkRedoDiarization()}
                    disabled={runningBulkRedo}
                    className="inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs rounded-lg border border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100 disabled:opacity-50"
                    title="Queue re-diarization across eligible channel episodes"
                >
                    {runningBulkRedo ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                    Bulk Re-diarize Channel
                </button>
            </div>
            <SpeakerList channelId={id} />
        </div>
    );
}
