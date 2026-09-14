class RetryableOperationError(RuntimeError):
    """Base error for transient failures that are safe to execute again."""

    error_code = "retryable_operation_error"
