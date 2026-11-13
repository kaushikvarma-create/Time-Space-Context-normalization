"""Typed errors. Library code raises these; only the CLI decides exit codes.

Boundary rule: catch the specific subclass you can handle (e.g. VLMError -> store the frame without VLM
fields). Anything else is a bug and must propagate to the top-level handler, which logs the traceback.
"""


class SbiError(Exception):
    """Base class for all expected, handled failures."""


class ConfigError(SbiError):
    """Missing or invalid configuration (settings file, space file, environment)."""


class CameraError(SbiError):
    """A frame could not be obtained from the camera/source."""


class DetectorError(SbiError):
    """The object detector failed to load or run."""


class SegmenterError(SbiError):
    """The segmenter (mask model) could not load or run. Frames are still processed, without masks."""


class VLMError(SbiError):
    """The scene-reading model failed after retries or returned an unusable answer."""


class StorageError(SbiError):
    """Database or filesystem failure (migration, transaction, image write)."""


class CurationError(SbiError):
    """The vocabulary curator's embedding or judge model failed."""
