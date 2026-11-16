"""Real loopback sockets: TIME_WAIT restart and active-owner preservation."""
import errno
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class PortReuseTests(unittest.TestCase):
    def policy(self, port, folder):
        p=Path(folder)/'policy.json'
        p.write_text(json.dumps({'schema':1,'port':port,'authentication':'disabled','listeners':[
            {'address':'127.0.0.1','port':port,'kind':'loopback','allowed_sources':['127.0.0.1/32']}]}))
        return p
    def check(self, policy):
        return subprocess.run([sys.executable,str(ROOT/'scripts/check_port.py'),'--policy',str(policy)],capture_output=True,text=True,timeout=5)
    def test_timewait_does_not_block_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            server=socket.socket();server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0));port=server.getsockname()[1];server.listen(1)
            client=socket.create_connection(('127.0.0.1',port));accepted,_=server.accept();server.close()
            accepted.shutdown(socket.SHUT_WR);self.assertEqual(client.recv(1),b'');client.close();accepted.close()
            with socket.socket() as ordinary:
                with self.assertRaises(OSError) as failed: ordinary.bind(('127.0.0.1',port))
                self.assertEqual(failed.exception.errno,errno.EADDRINUSE)
            result=self.check(self.policy(port,folder));self.assertEqual(result.returncode,0,result.stderr)
    def test_real_listener_is_not_bypassed_or_stopped(self):
        with tempfile.TemporaryDirectory() as folder, socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1];listener.listen(1)
            result=self.check(self.policy(port,folder));self.assertNotEqual(result.returncode,0)
            with socket.create_connection(('127.0.0.1',port),timeout=2): pass
