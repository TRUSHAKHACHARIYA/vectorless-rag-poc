class VectorlessRAGException(Exception):
    def __init__(self, message: str, details: dict | str | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class DocumentParseError(VectorlessRAGException):
    pass


class UnsupportedFileTypeError(VectorlessRAGException):
    pass


class FileTooLargeError(VectorlessRAGException):
    pass


class IndexNotFoundError(VectorlessRAGException):
    pass


class OllamaConnectionError(VectorlessRAGException):
    pass
