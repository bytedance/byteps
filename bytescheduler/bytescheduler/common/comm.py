from __future__ import absolute_import
import socket
import select
import threading
import pickle
import logging
import time
import os


def create_pickle_msg(rank, cmd, data):
    """Pickle the messages for transmission"""
    msg = {"rank": rank, "cmd": cmd, "data": data}
    msg = pickle.dumps(msg)
    return msg


class CommServer():
    """A communication server which broadcast messages to clients."""

    def __init__(self, rank, host, port, logger):
        """
        Arguments:
            rank: the rank of the worker.
            host: the IP of the server.
            port: the network port of the server.
            logger: the logging handler.
        """
        self._rank = rank
        self._host = host
        self._port = port
        self._logger = logger
        self._config = None
        self._exit = False

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind((host, port))
        self._sock.listen(5)
        self._rsocks = [self._sock]
        self._wsocks = []

        self._thread = threading.Thread(target=self._run, args=())
        self._thread.start()

    def _run(self):
        """Collect all client sockets and receive ADDNODE message."""
        while not self._exit:
            try:
                r_ready, w_ready, e_ready = select.select(self._rsocks, [], [], 10)
            except select.error as err:
                self._logger.error("Select error: {}".format(err))
                return
            if r_ready:
                for sd in r_ready:
                    if sd == self._sock:
                        fd, addr = self._sock.accept()
                        self._logger.debug('accept socket: {}, addr: {}'.format(fd, addr))
                        self._rsocks.append(fd)
                        self._wsocks.append(fd)
                    else:
                        msg = sd.recv(4096)
                        if not msg:
                            continue
                        msg = pickle.loads(msg)
                        self._logger.debug("Worker {} received message: {}".format(self._rank, msg))
                        rank = msg["rank"]
                        cmd = msg["cmd"]
                        if cmd == "ADDNODE":
                            self._logger.debug("Node {} is added.".format(rank))

    def shutdown(self):
        """Shut the server down."""
        self._exit = True
        self._thread.join()
        for sock in self._rsocks:
            sock.close()
        self._logger.debug("shutdown worker {}".format(self._rank))

    def broadcast(self, data):
        """ Broadcast a config (partition_size, credit_size) to all other workers.

        Arguments:
            data: the configuration to be sent to other workers.
        """
        self._config = data
        msg = create_pickle_msg(self._rank, "DATA", data)
        for sock in self._wsocks:
            sock.send(msg)

    def get(self):
        """Return the received configuration."""
        if self._config is not None:
            config = self._config
            self._config = None
            return config


class CommClient():
    """A communication client which registers to server and receives messages."""

    def __init__(self, rank, host, port, logger):
        """
        Arguments:
            rank: the rank of the worker.
            host: the IP of the server.
            port: the network port of the server.
            logger: the logging handler.
        """
        self._rank = rank
        self._host = host
        self._port = port
        self._logger = logger
        self._config = None
        self._exit = False

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self._thread = threading.Thread(target=self._run, args=())
        self._thread.start()

    def shutdown(self):
        """Stop thread and close socket."""
        self._exit = True
        self._thread.join()
        self._sock.close()
        self._logger.debug("shutdown worker {}".format(self._rank))

    def _run(self):
        """Listening to `CommServer` to get configuration."""
        while not self._exit:
            try:
                self._sock.connect((self._host, self._port))
                break
            except Exception as err:
                self._logger.debug("Error connect to rank 0: {}".format(err))
                time.sleep(1)

        # Register with server
        msg = create_pickle_msg(self._rank, "ADDNODE", "")
        self._sock.send(msg)

        while not self._exit:
            try:
                r_ready, w_ready, e_ready = select.select([self._sock], [], [], 10)
            except select.error as err:
                self._logger.error("Select error: {}".format(err))
                return

            if r_ready:
                msg = self._sock.recv(4096)
                if not msg:
                    continue
                msg = pickle.loads(msg)
                self._logger.debug("Received msg: {}".format(msg))
                cmd = msg["cmd"]
                if cmd == "DATA":
                    self._config = msg["data"]
                else:
                    self._logger.error("Received unknown cmd {}".format(cmd))

    def get(self):
        """Return the received configuration."""
        if self._config is not None:
            config = self._config
            self._config = None
            return config


def create_comm(rank, host=None, port=None, logger=None):
    """Launch a server in worker 0 and a client in other workers.

    Arguments:
        host: Network IP of worker 0. If not specified, use BYTESCHEDULER_ROOT_IP.
        port: Network port of worker 0. If not specified, use BYTESCHEDULER_ROOT_PORT.
    """
    if host is None:
        host = os.getenv("BYTESCHEDULER_ROOT_IP", "")
    if port is None:
        port = int(os.getenv("BYTESCHEDULER_ROOT_PORT", -1))
    assert host != "", "Unknown BYTESCHEDULER_ROOT_IP!"
    assert port >= 0, "Unknown BYTESCHEDULER_ROOT_PORT!"

    if logger is None:
        logger = logging.getLogger("ByteScheduler")
    logger.info("Comm host: {}, port: {}".format(host, port))

    if rank == 0:
        comm = CommServer(rank, host, port, logger)
    else:
        comm = CommClient(rank, host, port, logger)
    return comm
