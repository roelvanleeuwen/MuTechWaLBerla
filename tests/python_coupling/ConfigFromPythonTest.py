import waLBerla as wlb


class Scenario:
    def __init__(self):
        self.testInt = 4
        self.testString = "someString"
        self.testDouble = 4.43

    @wlb.member_callback
    def config(self):
        return {
            'DomainSetup': {
                'testInt': self.testInt,
                'testDouble': self.testDouble,
                'testString': self.testString,
            }
        }


scenarios = wlb.ScenarioManager()
scenarios.add(Scenario())