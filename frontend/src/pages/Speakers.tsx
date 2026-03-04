import { SpeakerList } from '../components/SpeakerList';

export function Speakers() {
    return (
        <div className="space-y-6">
            <h1 className="text-2xl font-bold text-slate-800">Speaker Manager</h1>
            <SpeakerList />
        </div>
    );
}
