"""Domain errors exposed by the inference engine."""


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
