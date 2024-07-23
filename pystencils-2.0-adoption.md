# Early Adoption of pystencils 2.0

You are viewing the development branch `pystencils2.0-adoption` for the early adoption of
[*pystencils 2.0*](https://i10git.cs.fau.de/pycodegen/pystencils/-/tree/v2.0-dev) into waLBerla.
Pystencils 2.0 is currently under development; this new version marks a fundamental redesign of pystencil's internals
and therefore requires extensive in-house testing.
Also, it already includes a number of new, often asked-for features, with more to come.
The documentation for pystencils 2.0 is currently available
[here](https://da15siwa.pages.i10git.cs.fau.de/dev-docs/pystencils-nbackend/).

If you wish to base your code-generation-based work within waLBerla on pystencils 2.0
and at the same time want to contribute to its testing and development, 
fork off this branch and set up your environment like this:

## Environment Setup

Clone waLBerla and switch to this feature branch.
Set up and activate a new python virtual environment, for example like:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages for code generation from the file [python/pystencils2.0-requirements.txt](python/pystencils2.0-requirements.txt).
This file currently includes the pointers to the pystencils 2.0 development branch and the associated feature branch in lbmpy;
this might change in the future, so regenerate your environment every once in a while.

```bash
pip install -r python/pystencils2.0-requirements.txt
```

Set up your build system with codegen activated and point CMake at the Python interpreter of your virtual environment:

```bash
mkdir build
cmake \
    -B build \
    -DWALBERLA_BUILD_WITH_PYTHON=1 \
    -DWALBERLA_BUILD_WITH_CODEGEN=1 \
    -DPython_EXECUTABLE=`pwd`/.venv/bin/python
```

And you're ready to go.

## Errors, Bugs, and Missing Features

As pystencils 2.0 is still under active development, you will notice that various features from pystencils 1.x
may now have brand-new bugs or be missing alltogether. We're working hard toward completing the feature set, so do not hesitate to report any bugs you find. Of course, bug fixes and feature contributions are also most welcome.
