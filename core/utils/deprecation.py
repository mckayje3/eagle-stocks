"""Deprecation warnings and version compatibility utilities"""

import warnings
import functools
from typing import Optional, Callable, Any


def deprecate(
    message: str,
    removal_version: Optional[str] = None,
    category: type = DeprecationWarning,
    stacklevel: int = 2
):
    """
    Issue a deprecation warning.

    Args:
        message: The deprecation message
        removal_version: Version when the feature will be removed
        category: Warning category (default: DeprecationWarning)
        stacklevel: Stack level for the warning

    Example:
        >>> deprecate("Use new_function() instead", removal_version="0.3.0")
    """
    if removal_version:
        full_message = f"{message} (will be removed in version {removal_version})"
    else:
        full_message = message

    warnings.warn(full_message, category, stacklevel=stacklevel)


def deprecated(
    reason: str,
    replacement: Optional[str] = None,
    removal_version: Optional[str] = None
) -> Callable:
    """
    Decorator to mark functions/classes as deprecated.

    Args:
        reason: Why this is deprecated
        replacement: What to use instead (optional)
        removal_version: When it will be removed (optional)

    Example:
        >>> @deprecated("Too slow", replacement="fast_function", removal_version="0.3.0")
        ... def old_function():
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            message = f"{func.__name__} is deprecated: {reason}"
            if replacement:
                message += f". Use {replacement} instead"
            if removal_version:
                message += f". Will be removed in version {removal_version}"

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper
    return decorator


def deprecated_argument(
    old_name: str,
    new_name: Optional[str] = None,
    removal_version: Optional[str] = None
):
    """
    Decorator to mark function arguments as deprecated.

    Args:
        old_name: Name of the deprecated argument
        new_name: Name of the new argument (optional)
        removal_version: When it will be removed (optional)

    Example:
        >>> @deprecated_argument("old_param", new_name="new_param", removal_version="0.3.0")
        ... def my_function(new_param=None, old_param=None):
        ...     if old_param is not None:
        ...         new_param = old_param
        ...     return new_param
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if old_name in kwargs:
                message = f"Argument '{old_name}' is deprecated"
                if new_name:
                    message += f", use '{new_name}' instead"
                if removal_version:
                    message += f". Will be removed in version {removal_version}"

                warnings.warn(message, DeprecationWarning, stacklevel=2)

                # Automatically map old to new if new_name is provided
                if new_name and new_name not in kwargs:
                    kwargs[new_name] = kwargs.pop(old_name)

            return func(*args, **kwargs)

        return wrapper
    return decorator


class DeprecatedClass:
    """
    Base class for deprecated classes.

    Example:
        >>> class OldModel(DeprecatedClass):
        ...     _deprecation_message = "Use NewModel instead"
        ...     _removal_version = "0.3.0"
    """
    _deprecation_message: str = "This class is deprecated"
    _removal_version: Optional[str] = None

    def __init__(self, *args, **kwargs):
        message = self._deprecation_message
        if self._removal_version:
            message += f". Will be removed in version {self._removal_version}"

        warnings.warn(message, DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


def warn_on_import(message: str, removal_version: Optional[str] = None):
    """
    Show a warning when a module is imported.

    Args:
        message: The warning message
        removal_version: Version when the module will be removed

    Example:
        In your deprecated module:
        >>> # At the top of the file
        >>> from .deprecation import warn_on_import
        >>> warn_on_import("This module is deprecated, use 'new_module' instead", "0.3.0")
    """
    if removal_version:
        full_message = f"{message} (will be removed in version {removal_version})"
    else:
        full_message = message

    warnings.warn(full_message, DeprecationWarning, stacklevel=2)
