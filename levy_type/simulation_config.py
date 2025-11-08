from attrs import define, field, validators
from attrs.validators import instance_of


@define(kw_only=True)
class SimulationConfig:
    T: float = field(validator=validators.and_(instance_of(float), validators.gt(0)), converter=float)
    N: int = field(validator=validators.and_(instance_of(int), validators.ge(0)))
    x_0: float = field(validator=instance_of(float), converter=float)
    alpha: float = field(
        validator=validators.and_(instance_of(float), validators.gt(0), validators.lt(2)), converter=float
    )
    random_seed: int = field(validator=instance_of(int))


@define(kw_only=True)
class DCSimulationConfig(SimulationConfig):
    h: float = field(validator=validators.and_(instance_of(float), validators.gt(0)))
    eps: float = field(validator=validators.and_(instance_of(float), validators.gt(0), validators.lt(1)))


@define(kw_only=True)
class ARSimulationConfig(SimulationConfig):
    delta: float = field(validator=validators.and_(instance_of(float), validators.gt(0)))
