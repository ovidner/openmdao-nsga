from copy import deepcopy

import hypothesis
import hypothesis.strategies as st
import pytest

from omnsga import VariableType
from omnsga.utils import (
    ConstraintDominatedFitness,
    ObjectiveValueWithConstraintViolation,
    disassemble_individuals,
    reassemble_individuals,
)

variable_type_strategy = st.sampled_from(VariableType)


@hypothesis.given(
    st.lists(
        st.tuples(variable_type_strategy, st.one_of(st.floats(), st.integers())),
        min_size=1,
    )
)
def test_individual_assembling(type_value_pairs):
    (types, values) = zip(*type_value_pairs)

    original_individual = list(values)
    reassembled_individual = [None] * len(values)
    (disassembled_individual,) = disassemble_individuals(types, [original_individual])
    reassemble_individuals(types, [(disassembled_individual, reassembled_individual)])

    assert reassembled_individual == original_individual


@st.composite
def objective_value_with_constraint_violation_st(draw, num_objectives):
    return ObjectiveValueWithConstraintViolation(
        objectives=tuple(
            draw(
                st.lists(
                    st.floats(allow_nan=False, allow_infinity=False),
                    min_size=num_objectives,
                    max_size=num_objectives,
                )
            )
        ),
        constraint_violation=draw(st.floats(min_value=0.0)),
    )


@hypothesis.given(
    st.integers(min_value=1, max_value=20).flatmap(
        lambda x: st.tuples(st.just(x), objective_value_with_constraint_violation_st(x))
    )
)
def test_constraint_dominated_fitness_deepcopy(num_objectives_values):
    num_objectives, values = num_objectives_values
    concrete_class = type(
        "Fitness", (ConstraintDominatedFitness,), {"weights": (-1,) * num_objectives}
    )
    fitness = concrete_class()
    fitness.values = values

    copied_fitness = deepcopy(fitness)
    assert fitness is not copied_fitness
    assert fitness.wvalues == copied_fitness.wvalues
    assert fitness.constraint_violation == copied_fitness.constraint_violation
