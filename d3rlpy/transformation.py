from .serializable_config import (
    DynamicConfig,
    generate_optional_config_generation,
)


class TransformationProtocol(DynamicConfig): ...


(
    register_transformation_callable,
    make_transformation_callable_field,
) = generate_optional_config_generation(TransformationProtocol)
