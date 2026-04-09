import os

os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/data/test_episode_clone_bootstrap.db")

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from src.db.database import Channel, TranscriptSegment, Video
from src.services import episode_clone as clone_svc


def _build_clone_test_session() -> tuple[object, Session, Channel, Video]:
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        SQLModel.metadata.create_all(conn)

    session = Session(engine)
    channel = Channel(url="https://example.com/@hoe-math", name="hoe_math")
    session.add(channel)
    session.commit()
    session.refresh(channel)

    video = Video(
        youtube_id="clone-test-video",
        channel_id=channel.id,
        title="ICK Diagnosis",
        description="Why people get the ick and what patterns trigger it.",
        view_count=8282752,
    )
    session.add(video)
    session.commit()
    session.refresh(video)

    session.add(
        TranscriptSegment(
            video_id=video.id,
            speaker_id=None,
            start_time=0.0,
            end_time=12.0,
            text="This episode breaks down common triggers behind the ick and why seemingly small behaviors can change attraction fast.",
        )
    )
    session.commit()
    return engine, session, channel, video


def test_normalize_clone_request_carries_engine_and_concepts():
    payload = clone_svc.normalize_clone_request(
        style_prompt="  Sharp explainer script  ",
        notes=" keep it broad ",
        provider_override="ChatGPT",
        model_override="gpt-4.1-mini",
        approved_concepts=["fear of neediness", "fear of neediness", "status shifts"],
        excluded_references=["Q&A session", "Q&A session", "hoe_math"],
    )

    assert payload["style_prompt"] == "Sharp explainer script"
    assert payload["provider_override"] == "openai"
    assert payload["model_override"] == "gpt-4.1-mini"
    assert payload["approved_concepts"] == ["fear of neediness", "status shifts"]
    assert payload["excluded_references"] == ["Q&A session", "hoe_math"]


def test_generate_episode_clone_suppresses_source_specific_contamination():
    _engine, session, _channel, video = _build_clone_test_session()

    result = clone_svc.generate_episode_clone(
        session,
        video_id=video.id,
        style_prompt="Create a clean, third-person explainer with no creator callbacks.",
        notes=None,
        approved_concepts=["small behaviors can abruptly change attraction", "framing matters more than intent"],
        excluded_references=["Q&A session", "hoe_math", "ICK Diagnosis"],
        text_generator=lambda _prompt: """
        {
          "suggested_title": "Why Attraction Can Collapse Fast",
          "opening_hook": "One habit can change the whole vibe.",
          "angle_summary": "Explains sudden attraction loss.",
          "originality_notes": ["fresh structure"],
          "script": "I mentioned in the Q&A session on hoe_math that the ICK Diagnosis pattern usually starts with neediness."
        }
        """,
        model_name="openai:gpt-4o-mini",
    )

    assert result["script"] == ""
    assert any("source-specific references" in warning.lower() for warning in result["warnings"])
    assert any("contamination" in note.lower() for note in result["originality_notes"])


def test_generate_episode_clone_includes_reference_evidence_under_review_constraints(monkeypatch):
    _engine, session, _channel, video = _build_clone_test_session()
    captured_prompt: dict[str, str] = {}

    monkeypatch.setattr(
        clone_svc,
        "_build_source_brief",
        lambda *_args, **_kwargs: "Source brief: the speaker explains why minor signals can rapidly shift attraction.",
    )
    monkeypatch.setattr(
        clone_svc,
        "_collect_related_context",
        lambda *_args, **_kwargs: (
            [
                {
                    "video_id": 999,
                    "title": "Status Signals",
                    "view_count": 1234,
                    "views_per_day": 45.6,
                    "semantic_score": 0.91,
                    "semantic_hit_count": 2,
                    "transcript_segment_count": 12,
                }
            ],
            [
                {
                    "chunk_id": 77,
                    "score": 0.88,
                    "video_id": 999,
                    "video_title": "Status Signals",
                    "speaker_name": "Narrator",
                    "start_time": 12.0,
                    "end_time": 24.0,
                    "chunk_text": "People react fast when confidence flips into approval-seeking behavior.",
                }
            ],
            [],
        ),
    )

    def fake_text_generator(prompt: str) -> str:
        captured_prompt["value"] = prompt
        return """
        {
          "suggested_title": "Why Small Signals Change Perception",
          "opening_hook": "One tiny shift can change the whole read.",
          "angle_summary": "Explains perception shifts using behavioral cues.",
          "originality_notes": ["Uses approved concepts with concrete support."],
          "script": "Small cues can quickly change how someone is perceived, especially when confidence turns into approval-seeking."
        }
        """

    result = clone_svc.generate_episode_clone(
        session,
        video_id=video.id,
        style_prompt="Write a crisp explainer with concrete examples.",
        notes="Keep the lesson grounded and specific.",
        approved_concepts=["small behaviors can rapidly change attraction", "approval-seeking can alter perceived status"],
        excluded_references=["Q&A session", "hoe_math", "ICK Diagnosis"],
        text_generator=fake_text_generator,
        model_name="openai:gpt-4o-mini",
    )

    prompt = captured_prompt["value"]
    assert "Approved concepts:" in prompt
    assert "Source episode evidence brief:" in prompt
    assert "Related semantic transcript evidence:" in prompt
    assert "People react fast when confidence flips into approval-seeking behavior." in prompt
    assert "Never cite or refer to the source evidence directly in the final script." in prompt
    assert result["script"]
