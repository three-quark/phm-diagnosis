# -*- coding: utf-8 -*-

import pika
import json
from base import DBInterface


class RabbitmqClient(DBInterface):

    def __init__(self, rabbitmqHost, rabbitmqPort, rabbitmqUsername, rabbitmqPassword, virtual_host, heartbeat, **kwargs):
        ''' build a credentials first, and then build the connection with rabbitmq '''
        credentials = pika.PlainCredentials(
            username=rabbitmqUsername, password=rabbitmqPassword)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=rabbitmqHost,
                                      port=rabbitmqPort,
                                      virtual_host=virtual_host,
                                      credentials=credentials,
                                      heartbeat=heartbeat))

    def write(self, exchange_name: list, queue_names: list, msg: dict):
        '''to publish a msg
        '''
        channel = self.connection.channel()
        channel.exchange_declare(
            exchange=exchange_name, exchange_type='fanout', durable=True)
        for queue_name in queue_names:
            channel.queue_declare(queue=queue_name, durable=True)
            channel.queue_bind(exchange=exchange_name, queue=queue_name)

            channel.basic_publish(exchange=exchange_name,
                                  routing_key=queue_name,
                                  body=json.dumps(msg).encode("utf-8"),
                                  properties=pika.BasicProperties(delivery_mode=2))

    def read(self):
        ''' force to implement by base interface '''
        pass
