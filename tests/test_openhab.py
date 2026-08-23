import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from openhab_voice_satellite.config import OpenHABConfig
from openhab_voice_satellite.openhab import OpenHABClient, OpenHABTimeoutError, make_session

from .fakes import FakeOpenHAB

# A throwaway self-signed CA, only ever handed to load_verify_locations so the
# ca_cert path can be asserted without a network peer. Never used to connect.
THROWAWAY_CA_PEM = """\
-----BEGIN CERTIFICATE-----
MIIDNzCCAh+gAwIBAgIUJ8ePkvG2gt/vLfrOJ9BhhpDutu8wDQYJKoZIhvcNAQEL
BQAwKjEoMCYGA1UEAwwfb3BlbmhhYi12b2ljZS1zYXRlbGxpdGUgdGVzdCBDQTAg
Fw0yNjA4MjMyMjUxMThaGA8yMTI2MDczMDIyNTExOFowKjEoMCYGA1UEAwwfb3Bl
bmhhYi12b2ljZS1zYXRlbGxpdGUgdGVzdCBDQTCCASIwDQYJKoZIhvcNAQEBBQAD
ggEPADCCAQoCggEBAMh8W3rIEb7V1gM06PA9qeBptDPZS7yB3f493h/NAshyO4/v
lq2e4k+3Dck5fslPBP6OlKv1FvwwZKRD5l/wxcYVnOXfs88ZTnY9KoNdn5h3mnFu
gxcOpDknd9tNMEsYUGKHreOska9/EVDr0f/QJqIBDFqNmb2Ts05ZsEAY5Exfnlbv
Cv47Db3c4aSriSRbi+o5GQHioe9ClOETn5+BZndsvZaGLzFzPZO8ZzZx8GxxLTU8
RRVZc7vjgUlLJ/7oy8okwP8F8OhcD3eoVbFTSbOe+KIzxIyXCDecVrbTX4JTYXLf
qncz5gmNJuk170SNSrbvNAYd8Sku8KPjIVdD5lECAwEAAaNTMFEwHQYDVR0OBBYE
FJzS38gba7+Ap10v+ig57lPdyT27MB8GA1UdIwQYMBaAFJzS38gba7+Ap10v+ig5
7lPdyT27MA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEBAFmumWOE
LN4zJ5EFHepCHIDNe589GuFgW2LRyTRi4d+6qmh0nTpk9uGW8uJ4wkFqYbxN43o+
pK4UDgdJ+2cZumvQPKikRqbAkQd3hOYhBw4ElK7J0EOj9Z2dy4teDeydXNZwGs1S
uX0qgh5k5wx2H+NMncSzJ9MziPz1VWGppYgio4Ql8wdC8lTKLQIW8+8guROY+u5r
C+S6QBX+F5rlayIlFVy0oBUzl8TF4apVeeK4An1p+7yMkr8srzm3gu6Tx0ApcFwv
arR4tBSvuYTqAalzkDGymHHlP0Ar/aiqRHmWsInd41K+a75Rseo9Ahy8Dy6S8/ds
Df5mYyoK8VctbmA=
-----END CERTIFICATE-----
"""


@pytest.fixture
async def fake_openhab():
    fake = FakeOpenHAB(response="Licht ist an.")
    server = TestServer(fake.build_app())
    await server.start_server(shutdown_timeout=0.2)
    yield fake, server
    # release before closing: a handler left mid-delay (the timeout tests)
    # keeps its connection open past the shutdown timeout, and the abandoned
    # server-side transport resurfaces later as an unraisable from another test
    fake.release()
    await server.close()


@pytest.fixture
async def session():
    session = aiohttp.ClientSession()
    yield session
    await session.close()


def _client(server: TestServer, session: aiohttp.ClientSession, **overrides) -> OpenHABClient:
    config = OpenHABConfig(url=str(server.make_url("")), **overrides)
    return OpenHABClient(config, session)


async def test_ping(fake_openhab, session):
    fake, server = fake_openhab
    await _client(server, session).ping()


async def test_command_and_response(fake_openhab, session):
    fake, server = fake_openhab
    response = await _client(server, session).send_command("schalte das licht an")
    assert response == "Licht ist an."
    assert fake.commands == ["schalte das licht an"]
    assert fake.llm_tools == ["item-send-command"]
    assert fake.conversations == [None]  # no ?conversation= without an id
    assert fake.headers[0]["Content-Type"].startswith("text/plain")


async def test_llm_tools_omitted_when_null(fake_openhab, session):
    fake, server = fake_openhab
    await _client(server, session, llm_tools=None).send_command("hallo")
    assert fake.llm_tools == [None]


async def test_conversation_param_sent(fake_openhab, session):
    fake, server = fake_openhab
    await _client(server, session).send_command("hallo", "abc-123")
    assert fake.conversations == ["abc-123"]
    assert fake.llm_tools == ["item-send-command"]  # composed with llmTools


async def test_end_conversation(fake_openhab, session):
    fake, server = fake_openhab
    await _client(server, session).end_conversation("abc-123")
    assert fake.deleted == ["abc-123"]


async def test_end_conversation_best_effort_on_http_error(fake_openhab, session, caplog):
    fake, server = fake_openhab
    fake.delete_status = 500
    await _client(server, session).end_conversation("abc-123")  # must not raise
    assert "conversation DELETE returned HTTP 500" in caplog.text


async def test_end_conversation_best_effort_on_connection_error(fake_openhab, session, caplog):
    fake, server = fake_openhab
    client = _client(server, session)
    await server.close()
    await client.end_conversation("abc-123")  # must not raise
    assert "failed to end conversation" in caplog.text


async def test_bearer_token_sent(fake_openhab, session, monkeypatch):
    fake, server = fake_openhab
    monkeypatch.setenv("OPENHAB_TOKEN", "secret-token")
    await _client(server, session).send_command("hallo")
    assert fake.headers[0]["Authorization"] == "Bearer secret-token"


async def test_http_error_raises(fake_openhab, session):
    fake, server = fake_openhab
    fake.status = 500
    with pytest.raises(aiohttp.ClientResponseError):
        await _client(server, session).send_command("hallo")


async def test_http_error_logs_body(fake_openhab, session, caplog):
    fake, server = fake_openhab
    fake.status = 400
    fake.error_body = '{"error":{"message":"Cannot interpret due to a technical problem."}}'
    with pytest.raises(aiohttp.ClientResponseError):
        await _client(server, session).send_command("hallo")
    assert "Cannot interpret" in caplog.text


async def test_response_timeout(fake_openhab, session):
    fake, server = fake_openhab
    fake.response_delay_s = 10
    # the dedicated type, not bare TimeoutError: the pipeline maps it to
    # "openHAB timed out" and must not confuse it with other timeouts
    with pytest.raises(OpenHABTimeoutError):
        await _client(server, session, response_timeout_s=0.2).send_command("hallo")


async def test_ping_proves_token_and_voice_endpoint(fake_openhab, session, monkeypatch):
    fake, server = fake_openhab
    monkeypatch.setenv("OPENHAB_TOKEN", "secret-token")
    await _client(server, session).ping()
    assert fake.ping_headers[0]["Authorization"] == "Bearer secret-token"


async def test_ping_raises_on_rejected_token(fake_openhab, session):
    fake, server = fake_openhab
    fake.ping_status = 401
    with pytest.raises(aiohttp.ClientResponseError):
        await _client(server, session).ping()


async def test_make_session_disables_ssl_verification(caplog):
    config = OpenHABConfig(verify_ssl=False)
    session = make_session(config)
    try:
        assert session.connector._ssl is False
        # the warning has to name the consequence: this is not "self-signed
        # certificates are ok", it is "the bearer token is unprotected"
        assert "any certificate" in caplog.text
        assert "API token is exposed" in caplog.text
    finally:
        await session.close()


async def test_ca_cert_authenticates_a_self_signed_server(tmp_path, caplog):
    """The secure alternative to verify_ssl: false.

    Without it a self-signed openHAB forced verification off entirely, which
    hands the bearer token to anyone who can intercept the connection.
    """
    import ssl as ssl_module

    ca = tmp_path / "ca.pem"
    ca.write_text(THROWAWAY_CA_PEM)
    session = make_session(OpenHABConfig(ca_cert=str(ca)))
    try:
        assert isinstance(session.connector._ssl, ssl_module.SSLContext)
        assert session.connector._ssl.verify_mode is ssl_module.CERT_REQUIRED
        assert session.connector._ssl.check_hostname is True
        assert "verification disabled" not in caplog.text
    finally:
        await session.close()


async def test_ca_cert_wins_over_verify_ssl_false(tmp_path):
    # a config carrying both is a half-finished migration off verify_ssl;
    # honoring the weaker one would silently keep the token exposed
    import ssl as ssl_module

    ca = tmp_path / "ca.pem"
    ca.write_text(THROWAWAY_CA_PEM)
    session = make_session(OpenHABConfig(ca_cert=str(ca), verify_ssl=False))
    try:
        assert isinstance(session.connector._ssl, ssl_module.SSLContext)
    finally:
        await session.close()


async def test_make_session_sets_default_timeout():
    # ping passes no per-request timeout; without a session default it
    # inherits aiohttp's 5-minute DEFAULT_TIMEOUT and stalls --check
    config = OpenHABConfig(response_timeout_s=7.5)
    session = make_session(config)
    try:
        assert session.timeout.total == 7.5
    finally:
        await session.close()


async def test_no_auth_header_without_token(fake_openhab, session, monkeypatch):
    # anonymous openHAB is a legitimate deployment: no token, no header
    fake, server = fake_openhab
    monkeypatch.delenv("OPENHAB_TOKEN", raising=False)
    await _client(server, session).send_command("hallo")
    assert "Authorization" not in fake.headers[0]


async def test_empty_body_returns_empty_string(fake_openhab, session):
    # an empty 200 is a silent round, not an error: the speakers all
    # return early on empty text
    fake, server = fake_openhab
    fake.response = ""
    assert await _client(server, session).send_command("hallo") == ""


async def test_non_ascii_round_trip(fake_openhab, session):
    fake, server = fake_openhab
    await _client(server, session).send_command("wie warm ist es in der Küche")
    assert fake.commands == ["wie warm ist es in der Küche"]
    assert "charset=utf-8" in fake.headers[0]["Content-Type"]


async def test_delete_404_logs_debug_not_warning(fake_openhab, session, caplog):
    # a barge-in before the server created the conversation is expected
    fake, server = fake_openhab
    fake.delete_status = 404
    with caplog.at_level("DEBUG"):
        await _client(server, session).end_conversation("abc-123")
    assert "conversation DELETE returned" not in caplog.text
    assert "unknown to server" in caplog.text


async def test_auth_error_names_the_token(fake_openhab, session, caplog):
    fake, server = fake_openhab
    fake.status = 401
    with pytest.raises(aiohttp.ClientResponseError):
        await _client(server, session).send_command("hallo")
    assert "OPENHAB_TOKEN" in caplog.text
