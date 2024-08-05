import waLBerla as wlb


class Scenario:
    def __init__(self, normal_dir, shift, periodicity):
        self.normal_dir = normal_dir
        self.shift = tuple(shift)
        self.periodicity = tuple(periodicity)

    @wlb.member_callback
    def config(self):
        normal_vector = [0] * 3
        normal_vector[self.normal_dir] = 1
        normal_vector = tuple(normal_vector)

        return {
            'DomainSetup': {
                'blocks': (3, 3, 3),
                'cellsPerBlock': (4, 4, 4),
                'cartesianSetup': 0,
                'periodic': self.periodicity,
            },
            'Boundaries': {
                'ShiftedPeriodicity': {
                    'shift': self.shift,
                    'normal': normal_vector
                }
            }
        }


scenarios = wlb.ScenarioManager()

for normal_dir in (0, 1, 2):
    for shift_dir in (0, 1, 2):
        if normal_dir == shift_dir:
            continue
        periodicity = 3 * [0]
        periodicity[shift_dir] = 1
        for shift_value in (2, 5, 8, 11):
            shift = [0] * 3
            shift[shift_dir] = shift_value
            scenarios.add(Scenario(normal_dir, shift, periodicity))
