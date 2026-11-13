"""Build an explicit localhost/Docker-IPAM listener policy from Docker inspect."""
import argparse
import ipaddress
import json
from pathlib import Path


def build(networks, port):
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError('Invalid API port')
    sources, default = [], None
    for network in networks:
        if network.get('Driver') != 'bridge':
            continue
        name = network['Name']
        interface = network.get('Options', {}).get('com.docker.network.bridge.name') or ('docker0' if name == 'bridge' else 'br-' + network['Id'][:12])
        if not Path('/sys/class/net', interface).is_dir():
            continue
        for config in network.get('IPAM', {}).get('Config') or []:
            if not config.get('Subnet') or not config.get('Gateway'):
                continue
            subnet = ipaddress.ip_network(config['Subnet'], strict=True)
            gateway = ipaddress.ip_address(config['Gateway'])
            if subnet.version != 4 or gateway.version != 4 or gateway not in subnet or subnet.prefixlen < 8:
                continue
            row = {'network': name, 'interface': interface, 'subnet': str(subnet), 'gateway': str(gateway)}
            sources.append(row)
            if name == 'bridge':
                default = row
    listeners = [{'address': '127.0.0.1', 'port': port, 'kind': 'loopback', 'allowed_sources': ['127.0.0.1/32']}]
    if default:
        listeners.append({'address': default['gateway'], 'gateway': default['gateway'], 'port': port,
                          'kind': 'docker-bridge', 'default_bridge': True,
                          'network': default['network'], 'interface': default['interface'],
                          'allowed_sources': sorted({source['subnet'] for source in sources})})
    return {'schema': 1, 'port': port, 'authentication': 'disabled', 'listeners': listeners, 'docker_sources': sources}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--networks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8005)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(json.loads(args.networks.read_text()), args.port), indent=2) + '\n')
