"""Check explicit API bindings before allocating GPUs; never stop another service."""
import argparse
import errno
import socket

from frontend import load_policy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default='/state/api-listeners.json')
    args = parser.parse_args()
    sockets = []
    try:
        for listener in load_policy(args.policy):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(sock)
            try:
                sock.bind((listener.address, listener.port))
            except OSError as exc:
                if listener.kind == 'docker-bridge' and exc.errno == errno.EADDRNOTAVAIL:
                    continue
                raise RuntimeError(f'API bind unavailable: {listener.address}:{listener.port}; no service was stopped') from exc
    finally:
        for sock in sockets:
            sock.close()
