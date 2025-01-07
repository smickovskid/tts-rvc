from flask_restx import fields

def define_models(api):
    """
    Define all the models for the application and return them as a dictionary.
    This allows reusing the models across multiple routes.
    """
    generate_request_model = api.model(
        "GenerateRequest",
        {
            "message": fields.String(
                required=True, description="Text message to generate audio"
            )
        },
    )
    generate_response_model = api.model(
        "GenerateResponse",
        {
            "error": fields.String(
                required=False, description="Errors"
            )
        },
    )

    health_response_model = api.model(
        "HealthResponse",
        {
            "status": fields.String(
                required=True, description="The health status of the service"
            )
        },
    )

    return {
        "generate_request_model": generate_request_model,
        "generate_response_model": generate_response_model,
        "health_response_model": health_response_model,
    }

