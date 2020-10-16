"""waLBerla callback decorators

This is the counterpart to the C++ class walberla::python_coupling::PythonCallback.
For details see documentation of this class.

This C++ class can call a Python function identified by a string using this C++ code::
    python_coupling::PythonCallback callback ( "someMagicString" );
    // expose some data here and pass them as arguments ( see C++ documentation for details)


There are two ways to mark a python function to be called by this callback.
The first option are normal function callbacks::

    @waLBerla.callback("someMagicString")
    def someArbitraryName( parameter1 ):
        pass


More advanced are callback classes, which can carry state::

    class MyScenario:
        def __init__( someState ):
            self._someState = someState

        @waLBerla.memberCallback
        def someMagicString:
            # react here according to the state
            pass

    scenarios = waLBerla.ScenarioManager()
    scenarios.add( MyScenario() )

Multiple instances of these callback classes can be added to the scenario manager,
which are then simulated after each other.


Internals:
^^^^^^^^^

The C++ waLBerla module walberla_cpp has a callbacks object.
To register a certain python function as callback it has to be set as attribute of this object:
``setattr( walberla_cpp.callbacks, "someMagicString", theCallbackFunction)``


"""

from __future__ import print_function, absolute_import, division, unicode_literals
import os
from functools import partial

try:
    import walberla_cpp
except ImportError:
    pass

# ---------------------------- Simple callback functions -----------------------------------------------------------

class callback:
    """Decorator class to mark a Python function as waLBerla callback"""

    def __init__(self, callbackFunction):
        if not type(callbackFunction) is str:
            raise Exception("waLBerla callback: Name of function has to be a string")
        self.callbackFunction = callbackFunction

    def __call__(self, f):
        try:
            from . import walberla_cpp
        except ImportError:
            try:
                import walberla_cpp
                setattr(walberla_cpp.callbacks, self.callbackFunction, f)
            except ImportError:
                # fail silently if waLBerla is not available
                pass

            return f


# ---------------------------- "Callback Classes"    -------------------------------------------------------------


def memberCallback(f):
    """Decorator to mark a member function as waLBerla callback"""
    f.waLBerla_callback_member = True
    return f


class ScenarioManager:
    """Use this class to simulate multiple scenarios
       A scenario is an instance of a class with member callbacks.
       See docstring of this module for an example.

       Internals:
           ScenarioManager is driven by "config" callbacks from the C++ code.
              ``for( auto configIt = python_coupling::configBegin(argc, argv); configIt != python_coupling::configEnd();
                     ++configIt )``
           Activation means to register the _configLoopCallback as 'config' waLBerla callback function
           which is called when a new scenario is expected.
           When config is called again the calbacks of the next scenario are activated.

    """
    def __init__(self):
        self._scenarios = []
        self._startScenario = 0

    def add(self, scenario):
        """Adds a scenario to the manager and activates the manager itself"""
        try:
            self._scenarios.append(scenario.config())
        except AttributeError:
            walberla_cpp.log_info_on_root("Scenario has no config function")
