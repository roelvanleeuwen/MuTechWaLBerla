#pragma once

#include <pybind11/pybind11.h>

namespace walberla {
namespace python_coupling {

namespace py = pybind11;

namespace detail {

template <typename T, py::return_value_policy Policy>
struct owning_iterator_state {
   owning_iterator_state(T _obj)
   : obj(_obj), it(obj.begin()), first_or_done(true) {}
   T obj;
   typename T::iterator it;
   bool first_or_done;
};

} // namespace detail

template <py::return_value_policy Policy = py::return_value_policy::reference_internal,
          typename T,
          typename... Extra>
py::iterator make_owning_iterator(T obj, Extra &&... extra) {
   using state = detail::owning_iterator_state<T, Policy>;

   if (!py::detail::get_type_info(typeid(state), false)) {
      py::class_<state>(py::handle(), "owning_iterator", py::module_local())
         .def("__iter__", [](state &s) -> state& { return s; })
         .def("__next__", [](state &s) -> typename T::value_type {
            if (!s.first_or_done)
               ++s.it;
            else
               s.first_or_done = false;
            if (s.it == s.obj.end()) {
               s.first_or_done = true;
               throw py::stop_iteration();
            }
            return *s.it;
         }, std::forward<Extra>(extra)..., Policy);
   }

   return cast(state(obj));
}

} // namespace python_coupling
} // namespace walberla
