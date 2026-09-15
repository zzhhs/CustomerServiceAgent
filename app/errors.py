class RetryableOperationError(RuntimeError):
    """Base error for transient failures that are safe to execute again."""

    error_code = "retryable_operation_error"


class MissingInputError(ValueError):
    """The current task can continue after the user supplies specific fields."""

    def __init__(self, message: str, *, fields: list[str]) -> None:
        super().__init__(message)
        self.fields = fields
