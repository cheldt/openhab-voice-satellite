import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from stt_proxy.config import OpenHABConfig
from stt_proxy.openhab import OpenHABClient

from .fakes import FakeOpenHAB


@pytest.fixture
async def fake_openhab():
    fake = FakeOpenHAB(response="Licht ist an.")
    server = TestServer(fake.build_app())
    await server.start_server(shutdown_timeout=0.2)
    yield fake, server
    await server.close()


async def _make_client(server: TestServer, **overrides) -> tuple[OpenHABClient, aiohttp.ClientSession]:
    config = OpenHABConfig(url=str(server.make_url("")), **overrides)
    session = aiohttp.ClientSession()
    client = OpenHABClient(config, session)
    return client, session


async def test_ping(fake_openhab):
    fake, server = fake_openhab
    client, session = await _make_client(server)
    try:
        await client.ping()
    finally:
        await session.close()


async def test_command_and_response(fake_openhab):
    fake, server = fake_openhab
    client, session = await _make_client(server)
    try:
        response = await client.send_command("schalte das licht an")
        assert response == "Licht ist an."
        assert fake.commands == ["schalte das licht an"]
        assert fake.llm_tools == ["item-send-command"]
        assert fake.headers[0]["Content-Type"].startswith("text/plain")
    finally:
        await session.close()


async def test_llm_tools_omitted_when_null(fake_openhab):
    fake, server = fake_openhab
    client, session = await _make_client(server, llm_tools=None)
    try:
        await client.send_command("hallo")
        assert fake.llm_tools == [None]
    finally:
        await session.close()


async def test_bearer_token_sent(fake_openhab, monkeypatch):
    fake, server = fake_openhab
    monkeypatch.setenv("OPENHAB_TOKEN", "secret-token")
    client, session = await _make_client(server)
    try:
        await client.send_command("hallo")
        assert fake.headers[0]["Authorization"] == "Bearer secret-token"
    finally:
        await session.close()


async def test_http_error_raises(fake_openhab):
    fake, server = fake_openhab
    fake.status = 500
    client, session = await _make_client(server)
    try:
        with pytest.raises(aiohttp.ClientResponseError):
            await client.send_command("hallo")
    finally:
        await session.close()


async def test_http_error_logs_body(fake_openhab, caplog):
    fake, server = fake_openhab
    fake.status = 400
    fake.error_body = '{"error":{"message":"Cannot interpret due to a technical problem."}}'
    client, session = await _make_client(server)
    try:
        with pytest.raises(aiohttp.ClientResponseError):
            await client.send_command("hallo")
        assert "Cannot interpret" in caplog.text
    finally:
        await session.close()


async def test_response_timeout(fake_openhab):
    fake, server = fake_openhab
    fake.response_delay_s = 10
    client, session = await _make_client(server, response_timeout_s=0.2)
    try:
        with pytest.raises(TimeoutError):
            await client.send_command("hallo")
    finally:
        await session.close()
