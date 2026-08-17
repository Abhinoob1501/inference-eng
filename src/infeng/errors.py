"""Transport-neutral failures raised below the HTTP layer.

FastAPI maps these classes to status codes in one place. Keeping that mapping out of
the engine lets Python callers handle the same failures without importing FastAPI.
"""


class InferenceEngineError(Exception):
    """Base class for expected engine failures."""


class InvalidRequestError(InferenceEngineError, ValueError):
    """The request cannot be served because its inputs are invalid."""


class EngineBusyError(InferenceEngineError):
    """No generation slot became available before the queue timeout."""


class EngineNotReadyError(InferenceEngineError):
    """The API process is alive but the model could not be loaded."""


class GenerationError(InferenceEngineError):
    """Model generation failed after the request was accepted."""
