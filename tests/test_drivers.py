import string
from functools import partial

import hypothesis
import hypothesis.extra.numpy as np_st
import hypothesis.strategies as st
import numpy as np
import openmdao.api as om
import pymop
import scipy as sp
import scop
from deap.tools import uniform_reference_points
from scop import DatasetRecorder

import omnsga
from omnsga import Nsga2Driver, Nsga3Driver, VariableType, add_design_var

ONES = np.ones((2,))
almost_equal = partial(np.allclose, rtol=1e-2, atol=1e-2)

vec_len = np.vectorize(len)

var_name_st = st.text(alphabet=string.ascii_letters, min_size=1)
var_type_st = st.sampled_from(VariableType)


class NoiseComponent(om.ExplicitComponent):
    def setup(self):
        self.add_output("y")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["y"] = np.random.random()


class PassthroughComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("shape")
        self.options.declare("in_type", types=VariableType)
        self.options.declare("out_type", types=VariableType)

    def setup(self):
        if self.options["in_type"] is VariableType.CONTINUOUS:
            self.add_input("in", shape=self.options["shape"])
        else:
            self.add_discrete_input("in", None)

        if self.options["out_type"] is VariableType.CONTINUOUS:
            self.add_output("out", shape=self.options["shape"])
        else:
            self.add_discrete_output("out", None)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inputs_ = (
            inputs
            if self.options["in_type"] is VariableType.CONTINUOUS
            else discrete_inputs
        )
        outputs_ = (
            outputs
            if self.options["out_type"] is VariableType.CONTINUOUS
            else discrete_outputs
        )

        outputs_["out"] = inputs_["in"]


@st.composite
def variable_st(draw):
    type_ = draw(var_type_st)
    shape = draw(np_st.array_shapes(min_dims=0))

    if type_.bounded:
        dtype, dtype_st, eps = (
            (np.int, st.integers, 1) if type_.discrete else (np.float, st.floats, 1e-6)
        )

        lower = draw(
            np_st.arrays(
                shape=shape,
                dtype=dtype,
                elements=dtype_st(max_value=1e9, min_value=-1e9),
            )
        )
        upper = lower + draw(
            np_st.arrays(
                shape=shape,
                dtype=dtype,
                elements=dtype_st(min_value=eps, max_value=1e3),
            )
        )

        output = {"lower": lower, "upper": upper}

    else:
        output = {
            "values": draw(
                np_st.arrays(
                    shape=shape,
                    dtype=object,
                    elements=st.sets(
                        st.floats(allow_nan=False)
                        if type_.ordered
                        else st.text(alphabet=string.ascii_letters),
                        min_size=1,
                        max_size=10,
                    ),
                )
            )
        }

    return {"type": type_, "shape": shape, **output}


# @hypothesis.reproduce_failure("5.6.0", b"AXicY2RkYGBkBBIIAAAAYgAG")
@hypothesis.settings(deadline=15000, max_examples=20, print_blob=True)
@hypothesis.given(
    variables=st.lists(
        st.tuples(var_name_st, variable_st()),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0],
    )
)
def test_variable_mixing(variables):
    prob = om.Problem()
    print("foo")

    for name, var in variables:
        group = prob.model.add_subsystem(name, om.Group())
        indeps = group.add_subsystem("indeps", om.IndepVarComp())
        if var["type"] is VariableType.CONTINUOUS:
            indeps.add_output("x", shape=var["shape"])
        else:
            indeps.add_discrete_output("x", None)

        add_design_var(group, "indeps.x", **var)

    noise = prob.model.add_subsystem("noise", NoiseComponent())
    noise.add_objective("y")

    recorder = DatasetRecorder()
    prob.driver = Nsga2Driver(
        termination_criterion=omnsga.MaxGenerationsCriterion(10),
        min_population_size=4,
        random_seed=0,
        verbose=False,
        # Cached iterations are not recorded
        use_cache=False,
    )
    prob.driver.add_recorder(recorder)

    try:
        prob.setup()
        prob.run_driver()
    finally:
        prob.cleanup()

    cases = recorder.assemble_dataset(prob.driver)

    expected_value_coverage = 0.66

    for name, var in variables:
        values = cases[f"{name}.indeps.x"].values
        if var["type"].bounded:
            upper = var["upper"]
            lower = var["lower"]

            assert np.all((lower <= values) & (values <= upper))
            assert np.all(
                np.ptp(values, axis=0) / (upper - lower) >= expected_value_coverage
            )
        else:
            unique_values = np.apply_along_axis(set, axis=0, arr=values)
            assert np.all(unique_values <= var["values"])

            assert np.all(
                vec_len(unique_values)
                >= np.ceil(vec_len(var["values"]) * expected_value_coverage)
            )


class PymopComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("problem", types=pymop.Problem)
        # self.options.declare("discrete_input", types=bool, default=False)

    def setup(self):
        problem = self.options["problem"]
        self.add_input("var", shape=(problem.n_var,))
        self.add_output("obj", shape=(problem.n_obj,))
        if problem.n_constr:
            self.add_output("con", shape=(problem.n_constr,))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        out = self.options["problem"].evaluate(
            inputs["var"], return_values_of=["F", "G"], return_as_dictionary=True
        )
        outputs["obj"] = out["F"]
        if self.options["problem"].n_constr:
            outputs["con"] = out["G"]


class PymopGroup(om.Group):
    def initialize(self):
        self.options.declare("problem", types=pymop.Problem)

    def setup(self):
        problem = self.options["problem"]
        self.add_subsystem(
            "indeps", om.IndepVarComp("var", shape=(problem.n_var,)), promotes=["*"]
        )
        self.add_subsystem(
            "problem", PymopComponent(problem=self.options["problem"]), promotes=["*"]
        )
        add_design_var(
            self, "var", shape=(problem.n_var,), lower=problem.xl, upper=problem.xu
        )
        self.add_objective("obj")
        if problem.n_constr:
            self.add_constraint("con", upper=0.0)


def test_unconstrained_dtlz1():
    recorder = scop.DatasetRecorder()
    pymop_problem = pymop.DTLZ1(n_var=3, n_obj=3)

    prob = om.Problem()
    prob.model = PymopGroup(problem=pymop_problem)
    prob.driver = Nsga3Driver(
        termination_criterion=omnsga.MaxGenerationsCriterion(20), random_seed=0
    )
    prob.driver.add_recorder(recorder)

    try:
        prob.setup()
        prob.run_driver()
    finally:
        prob.cleanup()

    cases = recorder.assemble_dataset(prob.driver)
    pareto_cases = scop.pareto_subset(cases)

    distance_function = "euclidean"
    ref_dirs = uniform_reference_points(pymop_problem.n_obj, p=4)
    ideal_pareto_front = pymop_problem.pareto_front(ref_dirs)
    min_pareto_point_distance = sp.spatial.distance.pdist(
        ideal_pareto_front, distance_function
    ).min()

    distances = sp.spatial.distance.cdist(
        pareto_cases["problem.obj"].values, ideal_pareto_front, distance_function
    )
    distances_to_ideal = np.min(distances, axis=0)

    assert distances_to_ideal.max() <= min_pareto_point_distance * 0.75
