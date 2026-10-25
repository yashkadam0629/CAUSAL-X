"""Relational metadata / results store (SQLAlchemy 2.x).

Bulk analytic tables (customers, loans, repayments, interventions, outcomes, economic indicators) live as Parquet files per dataset
version under data/ (columnar, reproducible, fingerprinted); this database stores their metadata and every derived result,
with foreign keys back to dataset and model versions so each result is traceable.
SQLite is the zero-install development default; PostgreSQL is selected by CAUSALX_DATABASE_URL. They are NOT identical
(e.g. SQLite does not enforce types or concurrent writers), see docs/architecture.md.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from ..core.config import settings


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(32))  # synthetic | upload
    scenario: Mapped[str | None] = mapped_column(String(40))
    n_customers: Mapped[int] = mapped_column(Integer)
    capabilities: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    row_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    storage_path: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    dataset = relationship("Dataset", back_populates="versions")


class DataQualityReport(Base):
    __tablename__ = "data_quality_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    score: Mapped[float] = mapped_column(Float)
    report: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # risk | cate
    name: Mapped[str] = mapped_column(String(120))
    target: Mapped[str] = mapped_column(String(60))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str | None] = mapped_column(String(500))
    seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    metrics = relationship("ModelMetric", back_populates="model", cascade="all, delete-orphan")


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_pk: Mapped[int] = mapped_column(ForeignKey("model_registry.id", ondelete="CASCADE"), index=True)
    split: Mapped[str] = mapped_column(String(20))
    metrics: Mapped[dict] = mapped_column(JSON)
    model = relationship("ModelRegistry", back_populates="metrics")


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"), index=True)
    kind: Mapped[str] = mapped_column(String(30))  # causal | design
    name: Mapped[str] = mapped_column(String(200))
    manifest: Mapped[dict] = mapped_column(JSON)  # treatment/outcome definitions, method, assumptions, seed, versions, hashes
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class CausalEstimateRow(Base):
    __tablename__ = "causal_estimates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_pk: Mapped[int | None] = mapped_column(ForeignKey("experiments.id", ondelete="SET NULL"), index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    treatment: Mapped[str] = mapped_column(String(40))
    comparison: Mapped[str] = mapped_column(String(40))
    outcome: Mapped[str] = mapped_column(String(40))
    estimator: Mapped[str] = mapped_column(String(40))
    ate: Mapped[float | None] = mapped_column(Float)
    ci_low: Mapped[float | None] = mapped_column(Float)
    ci_high: Mapped[float | None] = mapped_column(Float)
    n: Mapped[int | None] = mapped_column(Integer)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class CausalDiagnosticRow(Base):
    __tablename__ = "causal_diagnostics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    estimate_pk: Mapped[int] = mapped_column(ForeignKey("causal_estimates.id", ondelete="CASCADE"), index=True)
    diagnostics: Mapped[dict] = mapped_column(JSON)
    sensitivity: Mapped[dict | None] = mapped_column(JSON)


class UpliftResultRow(Base):
    __tablename__ = "uplift_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    model_pk: Mapped[int | None] = mapped_column(ForeignKey("model_registry.id", ondelete="SET NULL"))
    arm: Mapped[str] = mapped_column(String(40))
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    config: Mapped[dict] = mapped_column(JSON)
    summary: Mapped[dict] = mapped_column(JSON)
    comparison: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(40))
    policy_version: Mapped[str] = mapped_column(String(60))
    model_version: Mapped[str] = mapped_column(String(80))
    manifest: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    allocations = relationship("OptimizationAllocation", back_populates="run", cascade="all, delete-orphan")


class OptimizationAllocation(Base):
    __tablename__ = "optimization_allocations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_pk: Mapped[int] = mapped_column(ForeignKey("optimization_runs.id", ondelete="CASCADE"), index=True)
    customer_id: Mapped[str] = mapped_column(String(40), index=True)
    action: Mapped[str] = mapped_column(String(40))
    baseline_risk: Mapped[float] = mapped_column(Float)
    arr: Mapped[float] = mapped_column(Float)
    arr_se: Mapped[float] = mapped_column(Float)
    cost: Mapped[float] = mapped_column(Float)
    net_benefit: Mapped[float] = mapped_column(Float)
    run = relationship("OptimizationRun", back_populates="allocations")


class ScenarioRun(Base):
    __tablename__ = "scenario_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    config: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Policy(Base):
    __tablename__ = "policies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_id: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    definition: Mapped[dict] = mapped_column(JSON)
    version: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class PolicyEvaluationRow(Base):
    __tablename__ = "policy_evaluations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_pk: Mapped[int | None] = mapped_column(ForeignKey("policies.id", ondelete="SET NULL"), index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class DecisionRecommendation(Base):
    __tablename__ = "decision_recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decision_id: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    run_pk: Mapped[int | None] = mapped_column(ForeignKey("optimization_runs.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    customer_id: Mapped[str] = mapped_column(String(40), index=True)
    recommended_action: Mapped[str] = mapped_column(String(40))
    eligibility_result: Mapped[dict] = mapped_column(JSON)
    estimated_incremental_benefit: Mapped[float | None] = mapped_column(Float)
    estimated_intervention_cost: Mapped[float | None] = mapped_column(Float)
    expected_net_benefit: Mapped[float | None] = mapped_column(Float)
    effect_uncertainty: Mapped[float | None] = mapped_column(Float)
    policy_reason: Mapped[str] = mapped_column(Text)
    explanation: Mapped[dict] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(80))
    policy_version: Mapped[str] = mapped_column(String(60))
    data_version: Mapped[str] = mapped_column(String(64))
    human_review_required: Mapped[bool] = mapped_column(Boolean, default=False)
    human_review_status: Mapped[str] = mapped_column(String(30), default="pending_review")  # not_required|pending_review|approved|overridden|rejected
    recommendation_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (Index("ix_dec_status", "human_review_status"),)


class DecisionReview(Base):
    __tablename__ = "decision_reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decision_pk: Mapped[int] = mapped_column(ForeignKey("decision_recommendations.id", ondelete="CASCADE"), index=True)
    reviewer: Mapped[str] = mapped_column(String(120))
    outcome: Mapped[str] = mapped_column(String(30))
    original_action: Mapped[str] = mapped_column(String(40))
    final_action: Mapped[str] = mapped_column(String(40))
    reason: Mapped[str] = mapped_column(Text)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event: Mapped[str] = mapped_column(String(60), index=True)
    entity: Mapped[str] = mapped_column(String(60))
    entity_id: Mapped[str] = mapped_column(String(80))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)


class MonitoringSnapshot(Base):
    __tablename__ = "monitoring_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    snapshot: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|running|succeeded|failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


engine = create_engine(settings.database_url, future=True, connect_args={"check_same_thread": False} if settings.is_sqlite else {})
if settings.is_sqlite:
    @event.listens_for(engine, "connect")
    def _fk(dbapi_conn, _):  # SQLite needs foreign keys switched on explicitly (PostgreSQL always enforces them)
        dbapi_conn.execute("PRAGMA foreign_keys=ON")
        dbapi_conn.execute("PRAGMA journal_mode=WAL")

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
