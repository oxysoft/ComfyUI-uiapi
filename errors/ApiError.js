export class ApiError extends Error {
    constructor(message, statusCode, details) {
        super(message);
        this.statusCode = statusCode;
        this.details = details;
        this.name = 'ApiError';
    }
}
export class ConnectionError extends ApiError {
    constructor(message, details) {
        super(message, undefined, details);
        this.name = 'ConnectionError';
    }
}
export class ValidationError extends Error {
    constructor(message, field) {
        super(message);
        this.field = field;
        this.name = 'ValidationError';
    }
}
