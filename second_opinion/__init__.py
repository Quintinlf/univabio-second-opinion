"""Second Opinion — a diagnostic model that reports when not to trust it."""

from second_opinion.model import Assessment, Metrics, SecondOpinion
from second_opinion.references import REFERENCES, REFERENCES_BIBLIOGRAPHY

__all__ = [
    "Assessment",
    "Metrics",
    "REFERENCES",
    "REFERENCES_BIBLIOGRAPHY",
    "SecondOpinion",
]
