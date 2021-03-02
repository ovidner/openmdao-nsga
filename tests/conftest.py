import hypothesis
import numpy as np

np.seterr(all="warn")

hypothesis.settings.register_profile("fast", max_examples=5)
hypothesis.settings.register_profile("debugger", report_multiple_bugs=False)
