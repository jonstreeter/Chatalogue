import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_episode_chat_bootstrap.db")

from sqlmodel import Session, SQLModel, create_engine

from src.db.database import Channel, TranscriptChunkEmbedding, Video
from src.services import episode_chat as chat_svc


def test_episode_chat_answer_uses_selected_citations(monkeypatch):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    monkeypatch.setattr(
        chat_svc,
        "retrieve_episode_chat_citations",
        lambda *args, **kwargs: [
            {
                "chunk_id": 10,
                "segment_ids": [1, 2],
                "score": 0.91,
                "speaker_name": "Host",
                "start_time": 12.0,
                "end_time": 30.0,
                "support_text": "Poor sleep makes it harder to focus at work.",
            },
            {
                "chunk_id": 11,
                "segment_ids": [3],
                "score": 0.72,
                "speaker_name": "Guest",
                "start_time": 30.0,
                "end_time": 44.0,
                "support_text": "Sleep quality affects attention and memory.",
            },
        ],
    )

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@episode-chat", name="Episode Chat Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="episode-chat-video-1",
            channel_id=channel.id,
            title="Sleep and Attention",
            description="A discussion about sleep and focus.",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        answer = chat_svc.answer_episode_chat_question(
            session,
            video=video,
            prior_messages=[],
            latest_message="What does the episode say about sleep and focus?",
            scope_mode="episode",
            max_context_chunks=4,
            text_generator=lambda prompt: (
                '{"answer":"The episode says poor sleep reduces focus and attention.",'
                '"citation_indexes":[1],"grounding_note":"Supported directly by the cited transcript."}'
            ),
        )

    assert "poor sleep" in str(answer.get("answer") or "").lower()
    assert len(answer.get("citations") or []) == 1
    assert int((answer.get("citations") or [])[0]["chunk_id"]) == 10
    assert str(answer.get("semantic_query") or "").strip()


def test_episode_chat_answer_preserves_list_formatting(monkeypatch):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    monkeypatch.setattr(
        chat_svc,
        "retrieve_episode_chat_citations",
        lambda *args, **kwargs: [
            {
                "chunk_id": 10,
                "segment_ids": [1],
                "score": 0.91,
                "speaker_name": "Host",
                "start_time": 12.0,
                "end_time": 30.0,
                "support_text": "Point one. Point two.",
            },
        ],
    )

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@episode-chat-format", name="Episode Chat Format Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="episode-chat-video-format",
            channel_id=channel.id,
            title="Formatting Test",
            description="A discussion about formatting.",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        answer = chat_svc.answer_episode_chat_question(
            session,
            video=video,
            prior_messages=[],
            latest_message="Give me a bullet list.",
            scope_mode="episode",
            max_context_chunks=4,
            text_generator=lambda prompt: (
                '{"answer":"- First point\\n- Second point","citation_indexes":[1],"grounding_note":"Supported directly."}'
            ),
        )

    assert "- First point\n- Second point" in str(answer.get("answer") or "")


def test_episode_chat_coerces_bullet_format_when_requested(monkeypatch):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    monkeypatch.setattr(
        chat_svc,
        "retrieve_episode_chat_citations",
        lambda *args, **kwargs: [
            {
                "chunk_id": 21,
                "segment_ids": [4, 5],
                "score": 0.88,
                "speaker_name": "Host",
                "start_time": 8.0,
                "end_time": 28.0,
                "support_text": "First argument. Second argument. Third argument.",
            },
        ],
    )

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@episode-chat-bullets", name="Episode Chat Bullet Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="episode-chat-video-bullets",
            channel_id=channel.id,
            title="Bullet Format Test",
            description="A discussion about arguments.",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        answer = chat_svc.answer_episode_chat_question(
            session,
            video=video,
            prior_messages=[],
            latest_message="Give me a bullet point list of the core arguments.",
            scope_mode="episode",
            max_context_chunks=4,
            text_generator=lambda prompt: (
                '{"answer":"The core arguments include: First argument. Second argument. Third argument.",'
                '"citation_indexes":[1]}'
            ),
        )

    answer_text = str(answer.get("answer") or "")
    assert answer_text.startswith("- First argument.")
    assert "\n- Second argument." in answer_text


def test_episode_chat_coerces_bullets_with_inline_citation_markers(monkeypatch):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    monkeypatch.setattr(
        chat_svc,
        "retrieve_episode_chat_citations",
        lambda *args, **kwargs: [
            {
                "chunk_id": 31,
                "segment_ids": [10, 11],
                "score": 0.88,
                "speaker_name": "Host",
                "start_time": 8.0,
                "end_time": 28.0,
                "support_text": "First argument. Second argument. Third argument. Fourth argument.",
            },
        ],
    )

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@episode-chat-inline-citations", name="Episode Chat Inline Citation Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="episode-chat-video-inline-citations",
            channel_id=channel.id,
            title="Inline Citation Bullet Test",
            description="A discussion about arguments.",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        answer = chat_svc.answer_episode_chat_question(
            session,
            video=video,
            prior_messages=[],
            latest_message="Give me a bullet point list of the core arguments.",
            scope_mode="episode",
            max_context_chunks=4,
            text_generator=lambda prompt: (
                '{"answer":"The core arguments include: First argument.[6] Beliefs in the system persist. [3] The system depends on women''s loyalty and labor. [4] Women who defend patriarchal systems are responding to multiple pressures.",'
                '"citation_indexes":[1]}'
            ),
        )

    answer_text = str(answer.get("answer") or "")
    assert answer_text.startswith("- First argument.")
    assert "\n- Beliefs in the system persist." in answer_text
    assert "\n- The system depends on women" in answer_text


def test_episode_chat_broad_questions_expand_context_and_prompt(monkeypatch):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    captured: dict[str, object] = {}

    def fake_retrieve(*args, **kwargs):
        captured["limit"] = kwargs.get("limit")
        captured["broad_scope"] = kwargs.get("broad_scope")
        return [
            {
                "chunk_id": 41,
                "segment_ids": [1],
                "score": 0.92,
                "speaker_name": "Host",
                "start_time": 12.0,
                "end_time": 32.0,
                "support_text": "Argument one.",
            },
            {
                "chunk_id": 42,
                "segment_ids": [2],
                "score": 0.88,
                "speaker_name": "Host",
                "start_time": 36.0,
                "end_time": 56.0,
                "support_text": "Argument two.",
            },
        ]

    monkeypatch.setattr(chat_svc, "retrieve_episode_chat_citations", fake_retrieve)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@episode-chat-broad", name="Episode Chat Broad Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="episode-chat-video-broad",
            channel_id=channel.id,
            title="Broad Context Test",
            description="A discussion about several major arguments.",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        for idx in range(12):
            session.add(
                TranscriptChunkEmbedding(
                    channel_id=channel.id,
                    video_id=video.id,
                    start_time=float(idx * 60),
                    end_time=float(idx * 60 + 40),
                    chunk_text=f"Transcript chunk {idx + 1} covering argument {idx + 1}.",
                    chunk_token_estimate=20,
                    segment_ids_json="[]",
                    embedding_model="test",
                    embedding_dim=3,
                    embedding_bytes=b"123",
                    content_hash=f"hash-{idx}",
                )
            )
        session.commit()

        def fake_generator(prompt: str) -> str:
            captured["prompt"] = prompt
            return '{"answer":"- Argument one\\n- Argument two","citation_indexes":[1,2]}'

        answer = chat_svc.answer_episode_chat_question(
            session,
            video=video,
            prior_messages=[],
            latest_message="Give me a list of the top ten arguments in this episode.",
            scope_mode="episode",
            max_context_chunks=6,
            text_generator=fake_generator,
        )

    assert captured["broad_scope"] is True
    assert captured["limit"] == 10
    assert "Episode-wide context map:" in str(captured.get("prompt") or "")
    assert "Whole-episode synthesis mode: on" in str(captured.get("prompt") or "")
    assert "Transcript chunk 1 covering argument 1." in str(captured.get("prompt") or "")
    assert str(answer.get("prompt_version") or "") == "episode-chat-v2"


def test_episode_chat_context_map_samples_across_episode():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@episode-chat-map", name="Episode Chat Map Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="episode-chat-video-map",
            channel_id=channel.id,
            title="Context Map Test",
            description="A discussion about long-form structure.",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        for idx in range(20):
            session.add(
                TranscriptChunkEmbedding(
                    channel_id=channel.id,
                    video_id=video.id,
                    start_time=float(idx * 90),
                    end_time=float(idx * 90 + 45),
                    chunk_text=f"Section {idx + 1} of the episode.",
                    chunk_token_estimate=10,
                    segment_ids_json="[]",
                    embedding_model="test",
                    embedding_dim=3,
                    embedding_bytes=b"123",
                    content_hash=f"context-hash-{idx}",
                )
            )
        session.commit()

        context_map = chat_svc._build_episode_context_map(session, video=video, broad_scope=True)

    assert "Summary seed:" in context_map
    assert "Timeline map:" in context_map
    assert "Section 1 of the episode." in context_map
    assert "Section 20 of the episode." in context_map


def test_episode_chat_related_scope_returns_separate_related_citations(monkeypatch):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    monkeypatch.setattr(
        chat_svc,
        "retrieve_episode_chat_citations",
        lambda *args, **kwargs: [
            {
                "chunk_id": 51,
                "video_id": 100,
                "video_title": "Current Episode",
                "citation_scope": "episode",
                "segment_ids": [1],
                "score": 0.95,
                "speaker_name": "Host",
                "start_time": 15.0,
                "end_time": 35.0,
                "support_text": "Current episode argument.",
            },
        ],
    )
    monkeypatch.setattr(
        chat_svc,
        "retrieve_related_episode_chat_citations",
        lambda *args, **kwargs: [
            {
                "chunk_id": 61,
                "video_id": 200,
                "video_title": "Related Episode",
                "citation_scope": "related",
                "segment_ids": [7, 8],
                "score": 0.83,
                "speaker_name": "Host",
                "start_time": 42.0,
                "end_time": 60.0,
                "support_text": "Related-episode comparison point.",
            },
        ],
    )

    with Session(engine) as session:
        channel = Channel(url="https://example.com/@episode-chat-related", name="Episode Chat Related Test")
        session.add(channel)
        session.commit()
        session.refresh(channel)

        video = Video(
            youtube_id="episode-chat-video-related",
            channel_id=channel.id,
            title="Related Scope Test",
            description="A discussion with recurring themes.",
        )
        session.add(video)
        session.commit()
        session.refresh(video)

        answer = chat_svc.answer_episode_chat_question(
            session,
            video=video,
            prior_messages=[],
            latest_message="How does this compare to related episodes in the channel?",
            scope_mode="episode_related",
            max_context_chunks=6,
            text_generator=lambda prompt: (
                '{"answer":"The current episode makes the argument directly, and related episodes revisit it as part of a broader pattern.",'
                '"citation_indexes":[1],"related_citation_indexes":[1]}'
            ),
        )

    assert answer["scope_mode"] == "episode_related"
    assert answer["used_related_context"] is True
    assert len(answer["citations"]) == 1
    assert len(answer["related_citations"]) == 1
    assert answer["related_citations"][0]["video_title"] == "Related Episode"
    assert answer["related_video_ids"] == [200]
