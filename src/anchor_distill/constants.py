from __future__ import annotations

from enum import StrEnum


class JobType(StrEnum):
    TEACHER_LABEL = "teacher_label"
    BUILD_TEACHER_DATASET = "build_teacher_dataset"
    TRAIN_STUDENT = "train_student"
    EVALUATE_MODEL = "evaluate_model"
    BUILD_INDEX = "build_index"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATUSES = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}

ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.RUNNING: {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.SUCCEEDED: set(),
    JobStatus.FAILED: {JobStatus.QUEUED},
    JobStatus.CANCELLED: {JobStatus.QUEUED},
}
