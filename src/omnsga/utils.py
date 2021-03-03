import collections
import dataclasses
import enum
import functools
from copy import deepcopy
from itertools import chain

import deap
import numpy as np
from deap.base import Fitness


def listify(fn=None, wrapper=list):
    """
    From https://github.com/shazow/unstdlib.py/blob/master/unstdlib/standard/list_.py#L149

    A decorator which wraps a function's return value in ``list(...)``.

    Useful when an algorithm can be expressed more cleanly as a generator but
    the function should return an list.

    Example::

        >>> @listify
        ... def get_lengths(iterable):
        ...     for i in iterable:
        ...         yield len(i)
        >>> get_lengths(["spam", "eggs"])
        [4, 4]
        >>>
        >>> @listify(wrapper=tuple)
        ... def get_lengths_tuple(iterable):
        ...     for i in iterable:
        ...         yield len(i)
        >>> get_lengths_tuple(["foo", "bar"])
        (3, 3)
    """

    def listify_return(fn):
        @functools.wraps(fn)
        def listify_helper(*args, **kw):
            return wrapper(fn(*args, **kw))

        return listify_helper

    if fn is None:
        return listify_return
    return listify_return(fn)


@dataclasses.dataclass(frozen=True)
class VariableProperties:
    discrete: bool
    bounded: bool
    ordered: bool


class VariableType(VariableProperties, enum.Enum):
    CONTINUOUS = (False, True, True)
    INTEGER = (True, True, True)
    ORDINAL = (True, False, True)
    NOMINAL = (True, False, False)


@dataclasses.dataclass(frozen=True)
class ObjectiveValueWithConstraintViolation:
    objectives: tuple
    constraint_violation: float

    def __iter__(self):
        yield from self.objectives


class ConstraintDominatedFitness(Fitness):
    feasibility_tolerance = 1e-12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_violation = None

    def __deepcopy__(self, memo):
        # The base Fitness class uses an optimized deepcopy that throws away attributes
        copy_ = super().__deepcopy__(memo)
        copy_.constraint_violation = self.constraint_violation

        return copy_

    @property
    def feasible(self):
        return self.valid and self.constraint_violation <= self.feasibility_tolerance

    def set_values(self, values):
        if isinstance(values, ObjectiveValueWithConstraintViolation):
            self.constraint_violation = values.constraint_violation
            values = values.objectives
        Fitness.setValues(self, values)

    def del_values(self):
        self.constraint_violation = None
        Fitness.delValues(self)

    values = property(Fitness.getValues, set_values, del_values)

    def dominates(self, other, obj=slice(None)):
        if self.feasible and other.feasible:
            return super().dominates(other, obj)
        else:
            return self.constraint_violation < other.constraint_violation


class Individual(list):
    def __init__(self, *args, fitness_class, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness = fitness_class()

        # Metadata
        self.generation = None

    def __repr__(self):
        return f"Individual({super().__repr__()})"


@dataclasses.dataclass(frozen=True)
class IndividualBounds:
    lower: tuple
    upper: tuple

    @classmethod
    def from_design_var_meta(cls, design_var_meta):
        lower = {
            name: (meta["lower"] if meta["type"].bounded else np.zeros(meta["shape"]))
            for name, meta in design_var_meta.items()
        }
        upper = {
            name: (
                meta["upper"]
                if meta["type"].bounded
                else np.vectorize(len)(meta["values"]) - 1
            )
            for name, meta in design_var_meta.items()
        }

        return cls(
            lower=tuple(individual_sequence(lower, design_var_meta)),
            upper=tuple(individual_sequence(upper, design_var_meta)),
        )


def individual_sequence(design_vars, design_var_meta):
    return chain.from_iterable(
        np.broadcast_to(design_vars[name], meta["shape"]).flat
        for name, meta in design_var_meta.items()
    )


def individual_types_sequence(design_var_meta):
    return chain.from_iterable(
        [meta["type"]] * np.product(meta["shape"] or (1,))
        for meta in design_var_meta.values()
    )


def stretch_array(array, shape):
    try:
        return np.broadcast_to(array, shape)
    except ValueError:
        return np.reshape(array, shape)

    missing_dims = len(shape) - array.ndim
    indexer = (...,) + (np.newaxis,) * missing_dims

    array = array[indexer]

    return np.broadcast_to(array, shape)


def random_ints(shape, lower, upper):
    ret = np.empty(shape, dtype=np.int)

    lower = np.broadcast_to(lower, shape)
    upper = np.broadcast_to(upper, shape)

    for i in np.ndindex(*shape):
        ret[i] = np.random.randint(lower[i], upper[i], dtype=np.int)

    return ret


def random_floats(shape, lower, upper):
    ret = np.empty(shape)

    lower = stretch_array(lower, shape)
    upper = stretch_array(upper, shape)

    for i in np.ndindex(*shape):
        ret[i] = np.random.rand() * (upper[i] - lower[i]) + lower[i]

    return ret


def random_choice(shape: tuple, values: set):
    ret = np.empty(shape, dtype=object)

    values = np.broadcast_to(values, shape)

    for i in np.ndindex(*shape):
        ret[i] = np.random.choice(list(values[i]))

    return ret


def convert_design_vars_to_individual(design_var_meta, fitness_class, design_vars):
    """
    Converts a dict of OpenMDAO design variables into a DEAP individual.
    """
    return Individual(
        individual_sequence(design_vars, design_var_meta), fitness_class=fitness_class
    )


def convert_individual_to_design_vars(
    individual, design_var_meta, discrete_value_mappings
):
    """
    Converts a DEAP individual into a dict of OpenMDAO design variables.
    """
    ind = deepcopy(individual)

    design_vars = {}
    for name, meta in design_var_meta.items():
        shape = meta["shape"]
        type_ = meta["type"]
        ind_items = np.prod(shape, dtype=int)
        values = np.array(
            ind[:ind_items],
            dtype=(np.float if type_ is VariableType.CONTINUOUS else np.int),
        ).reshape(shape)

        if type_.discrete and not type_.bounded:
            values = values.astype("O")
            for arr_index, index in np.ndenumerate(values):
                a = discrete_value_mappings[name][arr_index][index]
                values[arr_index] = a
        assert values.shape == shape
        design_vars[name] = values if shape else values.item()
        ind = ind[ind_items:]

    return design_vars


def random_individual_value(type_, lower, upper):
    if type_ is VariableType.CONTINUOUS:
        return np.random.rand() * (upper - lower) + lower
    else:
        return np.random.randint(lower, upper + 1)


def init_population(count, individual_types, individual_bounds, fitness_class):
    for i in range(count):
        yield Individual(
            [
                random_individual_value(type_, lower, upper)
                for (type_, lower, upper) in zip(
                    individual_types, individual_bounds.lower, individual_bounds.upper
                )
            ],
            fitness_class=fitness_class,
        )


def disassemble_individuals(types, individuals):
    for ind in individuals:
        yield {
            type_: [attr for attr, attr_type in zip(ind, types) if attr_type is type_]
            for type_ in VariableType
        }


def types_index(types):
    counter = collections.Counter()
    for type_ in types:
        yield counter[type_]
        counter[type_] += 1


def reassemble_individuals(types, individual_pairs):
    for (index, (type_, sub_index)) in enumerate(zip(types, types_index(types))):
        for ind, targ_ind in individual_pairs:
            targ_ind[index] = ind[type_][sub_index]


def mate_disassembled(
    ind1,
    ind2,
    individual_types,
    individual_bounds,
    cont_eta=30,
    int_eta=30,
    ord_indpb=1.0,
    nom_indpb=1.0,
):
    (da_ind1, da_ind2, da_lower, da_upper) = disassemble_individuals(
        individual_types, (ind1, ind2, individual_bounds.lower, individual_bounds.upper)
    )

    deap.tools.cxSimulatedBinaryBounded(
        da_ind1[VariableType.CONTINUOUS],
        da_ind2[VariableType.CONTINUOUS],
        eta=cont_eta,
        low=da_lower[VariableType.CONTINUOUS],
        up=da_upper[VariableType.CONTINUOUS],
    )

    deap.tools.cxSimulatedBinaryBounded(
        da_ind1[VariableType.INTEGER],
        da_ind2[VariableType.INTEGER],
        eta=int_eta,
        low=da_lower[VariableType.INTEGER],
        up=da_upper[VariableType.INTEGER],
    )
    da_ind1[VariableType.INTEGER] = np.round(da_ind1[VariableType.INTEGER]).tolist()
    da_ind2[VariableType.INTEGER] = np.round(da_ind2[VariableType.INTEGER]).tolist()

    deap.tools.cxUniform(
        da_ind1[VariableType.ORDINAL], da_ind2[VariableType.ORDINAL], indpb=ord_indpb
    )

    deap.tools.cxUniform(
        da_ind1[VariableType.NOMINAL], da_ind2[VariableType.NOMINAL], indpb=nom_indpb
    )

    reassemble_individuals(individual_types, [(da_ind1, ind1), (da_ind2, ind2)])

    return ind1, ind2


def mutate_disassembled(
    ind,
    individual_types,
    individual_bounds,
    cont_eta=20,
    cont_indpb=1.0,
    int_eta=20,
    int_indpb=1.0,
    ord_indpb=1.0,
    nom_indpb=1.0,
):
    (da_ind, da_lower, da_upper) = disassemble_individuals(
        individual_types, (ind, individual_bounds.lower, individual_bounds.upper)
    )

    deap.tools.mutPolynomialBounded(
        da_ind[VariableType.CONTINUOUS],
        eta=cont_eta,
        indpb=cont_indpb,
        low=da_lower[VariableType.CONTINUOUS],
        up=da_upper[VariableType.CONTINUOUS],
    )

    deap.tools.mutPolynomialBounded(
        da_ind[VariableType.INTEGER],
        eta=int_eta,
        indpb=int_indpb,
        low=da_lower[VariableType.INTEGER],
        up=da_upper[VariableType.INTEGER],
    )
    da_ind[VariableType.INTEGER] = np.round(da_ind[VariableType.INTEGER]).tolist()

    deap.tools.mutUniformInt(
        da_ind[VariableType.ORDINAL],
        indpb=ord_indpb,
        low=da_lower[VariableType.ORDINAL],
        up=da_upper[VariableType.ORDINAL],
    )

    deap.tools.mutUniformInt(
        da_ind[VariableType.NOMINAL],
        indpb=nom_indpb,
        low=da_lower[VariableType.NOMINAL],
        up=da_upper[VariableType.NOMINAL],
    )

    reassemble_individuals(individual_types, [(da_ind, ind)])

    return (ind,)


def _make_discrete_value_mapping(values_set, type_):
    if type_.ordered:
        values_set = sorted(values_set)
    enum_values = enumerate(values_set)
    return {index: value for index, value in enum_values}


def make_discrete_value_mappings(design_var_meta):
    for name, meta in design_var_meta.items():
        type_ = meta["type"]
        if type_.bounded:
            continue

        yield name, np.vectorize(lambda x: _make_discrete_value_mapping(x, type_))(
            meta["values"]
        )


def epsilonify(value, eps=np.finfo(float).eps):
    value = np.copy(value)
    if isinstance(value, np.ndarray):
        value[value == 0.0] = eps
    else:
        value = value or eps
    return value


def constraint_violation(values, meta):
    keys = set(values.keys())
    assert keys == set(meta.keys())

    total_violation = 0.0
    for key in keys:
        var_meta = meta[key]
        val = values[key]

        lower = var_meta["lower"]
        lower_eps = epsilonify(lower)
        upper = var_meta["upper"]
        upper_eps = epsilonify(upper)
        equals = var_meta["equals"]
        equals_eps = epsilonify(equals)

        total_violation += np.sum(np.abs(np.fmax(upper, val) / upper_eps - 1))
        total_violation += np.sum(np.abs(np.fmin(lower, val) / lower_eps - 1))
        if equals is not None:
            total_violation += np.sum(np.abs(val / equals_eps - 1))

    assert total_violation >= 0.0
    return total_violation


def add_design_var(
    sys,
    name,
    *args,
    type=VariableType.CONTINUOUS,
    values=None,
    shape=(1,),
    **kwargs,
):
    if type.bounded and values:
        kwargs["lower"] = values[0]
        kwargs["upper"] = values[1]
        values = None

    if type.bounded and type.discrete:
        lower_int = kwargs.pop("lower", None)
        upper_int = kwargs.pop("upper", None)

    sys.add_design_var(name, *args, **kwargs)

    if sys._static_mode:
        design_vars = sys._static_design_vars
    else:
        design_vars = sys._design_vars

    # FIXME: Hacky McHackface
    if type.bounded and type.discrete:
        design_vars[name]["lower"] = np.broadcast_to(lower_int, shape)
        design_vars[name]["upper"] = np.broadcast_to(upper_int, shape)

    design_vars[name]["type"] = type
    design_vars[name]["values"] = values
    design_vars[name]["shape"] = shape
