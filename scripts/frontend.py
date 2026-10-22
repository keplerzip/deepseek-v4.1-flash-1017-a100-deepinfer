"""TCP peer ACL and byte-preserving forwarding to the isolated engine UDS.

Access follows R3.9: localhost plus the default Docker bridge gateway,
without API keys. All verified Docker bridge source subnets may use that gateway.
"""
import argparse
import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
import errno
import ipaddress
import json
from pathlib import Path
import socket


@dataclass(frozen=True)
class Listener:
    address: str
    port: int
    sources: tuple
    kind: str

    def permits(self, peer):
        try:
            address = ipaddress.ip_address(peer)
        except ValueError:
            return False
        return address.version == 4 and any(address in network for network in self.sources)


def load_policy(path):
    policy = json.loads(Path(path).read_text())
    port = policy.get('port')
    if policy.get('schema') != 1 or type(port) is not int or not 1024 <= port <= 65535 or policy.get('authentication') != 'disabled':
        raise ValueError('Unsupported API policy; expected no-auth explicit-port schema1')
    rows = policy.get('listeners')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2:
        raise ValueError('Expected localhost and at most one default Docker gateway')
    listeners, seen = [], set()
    for row in rows:
        address = ipaddress.ip_address(row['address'])
        if address.version != 4 or str(address) in seen or row['port'] != port:
            raise ValueError('IPv4 explicit bind addresses and one explicit port required')
        sources = tuple(ipaddress.ip_network(text, strict=True) for text in row['allowed_sources'])
        if not sources or any(source.version != 4 or source.prefixlen < 8 for source in sources):
            raise ValueError('Invalid IPv4 source allowlist')
        if row['kind'] == 'loopback':
            if str(address) != '127.0.0.1' or tuple(map(str, sources)) != ('127.0.0.1/32',):
                raise ValueError('Loopback API must bind exactly 127.0.0.1')
        elif row['kind'] == 'docker-bridge':
            if address.is_loopback or address.is_multicast or address.is_unspecified or address.is_link_local or address.is_reserved:
                raise ValueError('Invalid Docker API gateway address')
            if row.get('default_bridge') is not True or str(address) != row.get('gateway'):
                raise ValueError('Only the default Docker gateway is an API entry')
            verified = policy.get('docker_sources', [])
            verified_cidrs = {item['subnet'] for item in verified}
            if set(map(str, sources)) != verified_cidrs or not any(
                    item['gateway'] == str(address) and item['interface'] == row['interface'] and
                    item['network'] == row['network'] for item in verified):
                raise ValueError('Docker source allowlist must match verified Docker IPAM sources')
            if not any(address in source for source in sources):
                raise ValueError('Gateway outside Docker source subnets')
        else:
            raise ValueError('Unknown API listener kind')
        seen.add(str(address))
        listeners.append(Listener(str(address), port, sources, row['kind']))
    if '127.0.0.1' not in seen:
        raise ValueError('Localhost listener is required')
    return listeners


async def close_writer(writer):
    writer.close()
    try:
        await writer.wait_closed()
    except (OSError, ConnectionError):
        pass


def make_handler(listener, path):
    async def handle(reader, writer):
        peer = writer.get_extra_info('peername')
        if not peer or not listener.permits(peer[0]):
            writer.write(b'HTTP/1.1 403 Forbidden\r\nConnection: close\r\nContent-Length: 0\r\n\r\n')
            try:
                await writer.drain()
            except (OSError, ConnectionError):
                pass
            await close_writer(writer)
            return
        upstream_writer = None
        tasks = []
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(path)), timeout=10)

            async def copy(source, destination):
                while data := await asyncio.wait_for(source.read(65536), timeout=7200):
                    destination.write(data)
                    await destination.drain()
                if destination.can_write_eof():
                    destination.write_eof()
                    await destination.drain()

            tasks = [asyncio.create_task(copy(reader, upstream_writer)),
                     asyncio.create_task(copy(upstream_reader, writer))]
            await asyncio.gather(*tasks)
        except (OSError, ConnectionError, asyncio.TimeoutError):
            pass
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await close_writer(writer)
            if upstream_writer:
                await close_writer(upstream_writer)
    return handle


async def bind_listeners(listeners, path):
    """Bind exact IPv4 addresses. A removed bridge fails closed, not wildcard."""
    servers = []
    try:
        for listener in listeners:
            try:
                server = await asyncio.start_server(make_handler(listener, path), listener.address,
                                                    listener.port, family=socket.AF_INET)
            except OSError as error:
                if listener.kind == 'docker-bridge' and error.errno == errno.EADDRNOTAVAIL:
                    print(json.dumps({'api_listener': listener.address, 'status': 'SKIPPED_NOT_ASSIGNED',
                                      'port': listener.port}), flush=True)
                    continue
                raise
            servers.append(server)
            print(json.dumps({'api_listener': listener.address, 'status': 'LISTENING', 'port': listener.port,
                              'authentication': 'disabled',
                              'allowed_sources': list(map(str, listener.sources))}), flush=True)
    except BaseException:
        for server in servers:
            server.close()
            await server.wait_closed()
        raise
    return servers


async def serve(args):
    listeners = load_policy(args.policy)
    path = Path(args.socket)
    if not path.is_socket():
        raise ValueError('Expected the engine Unix socket')
    servers = await bind_listeners(listeners, path)
    async with AsyncExitStack() as stack:
        for server in servers:
            await stack.enter_async_context(server)
        await asyncio.gather(*(server.serve_forever() for server in servers))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', default='/state/api.sock')
    parser.add_argument('--policy', default='/state/api-listeners.json')
    asyncio.run(serve(parser.parse_args()))
