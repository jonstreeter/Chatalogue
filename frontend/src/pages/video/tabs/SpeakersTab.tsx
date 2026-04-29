import { Loader2 } from 'lucide-react';
import { SpeakerList } from '../../../components/SpeakerList';
import { useSpeakersTabStore } from '../../../store/useSpeakersTabStore';
import type { Speaker, TranscriptSegment, Video } from '../../../types';

type Props = {
    video: Video | null;
    videoId: number;
    isActive: boolean;
    onSegmentsUpdated: (updater: (segments: TranscriptSegment[]) => TranscriptSegment[]) => void;
    onSegmentsLoaded: (segments: TranscriptSegment[]) => void;
};

export function SpeakersTab({ video, videoId, isActive, onSegmentsUpdated, onSegmentsLoaded }: Props) {
    const handleSpeakerUpdated = useSpeakersTabStore((s) => s.handleSpeakerUpdated);
    const handleSpeakerMerged = useSpeakersTabStore((s) => s.handleSpeakerMerged);

    const updateSegmentsForSpeaker = (speakerId: number, name: string) => {
        onSegmentsUpdated((segments) =>
            segments.map((segment) =>
                segment.speaker_id === speakerId ? { ...segment, speaker: name } : segment
            )
        );
    };

    const onSpeakerUpdated = (updatedSpeaker: Speaker) => {
        handleSpeakerUpdated(updatedSpeaker, updateSegmentsForSpeaker);
    };

    const onSpeakerMerged = () => {
        void handleSpeakerMerged(videoId, onSegmentsLoaded);
    };

    if (!isActive) return null;

    return (
        <div className="h-full overflow-y-auto p-4">
            {video ? (
                <SpeakerList
                    videoId={video.id}
                    channelId={undefined}
                    onSpeakerUpdated={onSpeakerUpdated}
                    onSpeakerMerged={onSpeakerMerged}
                />
            ) : (
                <Loader2 className="animate-spin" />
            )}
        </div>
    );
}
