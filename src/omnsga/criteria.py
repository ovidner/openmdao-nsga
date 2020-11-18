import dataclasses

from deap.tools import Logbook

__all__ = ["MaxGenerationsCriterion", "MaxEvaluationsCriterion"]


class Criterion:
    def __call__(self, logbook: Logbook, population: list) -> bool:
        """
        The semantics of calling a Criterion is "is the criterion met?".
        """
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class MaxGenerationsCriterion(Criterion):
    max_generations: int

    def __call__(self, logbook, population):
        return logbook.select("gen")[-1] >= self.max_generations


@dataclasses.dataclass(frozen=True)
class MaxEvaluationsCriterion(Criterion):
    max_evaluations: int

    def __call__(self, logbook, population):
        return sum(logbook.select("nevals")) >= self.max_evaluations
