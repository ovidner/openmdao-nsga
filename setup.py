from setuptools import find_packages, setup

setup(
    name="openmdao-nsga",
    use_scm_version=True,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8, <4",
    setup_requires=["setuptools_scm"],
    install_requires=["openmdao>=3.4", "numpy", "deap"],
)
