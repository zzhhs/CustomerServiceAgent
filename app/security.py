from datetime import UTC, datetime

import jwt
from jwt import InvalidTokenError


class AuthenticationError(ValueError):
    pass


class JwtBearerAuthenticator:
    """Validate short-lived HS256 JWTs at the standalone API boundary."""

    def __init__(self, *, secret: str, issuer: str, audience: str) -> None:
        self._secret = secret
        self._issuer = issuer
        self._audience = audience

    def authenticate(self, authorization: str | None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError("缺少 Bearer 身份令牌。")
        token = authorization.removeprefix("Bearer ").strip()
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=["HS256"],
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except InvalidTokenError as exc:
            raise AuthenticationError("身份令牌无效或已经过期。") from exc
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise AuthenticationError("身份令牌缺少有效用户标识。")
        return subject

    @staticmethod
    def issue(
        *,
        user_id: str,
        expires_at: datetime,
        secret: str,
        issuer: str,
        audience: str,
    ) -> str:
        now = datetime.now(UTC)
        return jwt.encode(
            {
                "sub": user_id,
                "iat": now,
                "exp": expires_at,
                "iss": issuer,
                "aud": audience,
            },
            secret,
            algorithm="HS256",
        )
