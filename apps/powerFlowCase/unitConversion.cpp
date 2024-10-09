#include "core/Environment.h"

#include <cmath>
#include <iostream>
using namespace std;
using namespace walberla;

namespace walberla
{

struct physicalUnits
{
   real_t airfoilChordLength;
   real_t velocityMagnitude;
   real_t kinViscosity;
   real_t temperature;
   real_t rho;
};

} // namespace walberla
