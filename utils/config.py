import sys
from logging import Logger

def check_env_vars(env_vars: dict, logger: Logger):
    """Checks if all the keys in the dictionary have non-empty values."""
    missing_vars = []

    for key, value in env_vars.items():
        if not value:
            missing_vars.append(key)

    if missing_vars:
        logger.error(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    logger.info("All required environment variables are set.")

def get(env_vars: dict, key: str) -> str:
    """Helper function to get the value of an environment variable from a given dictionary."""
    return env_vars.get(key, "")
