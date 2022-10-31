# -*- coding: utf-8 -*-

import pika
import json
from .base import DBInterface


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
        self.channel = self.connection.channel()

    def write(self, exchange_name: str, exchange_type: str, queue_names: list, msg: dict):
        '''to publish a msg
        '''
        print(exchange_name, exchange_type, queue_names, msg)
        self.channel.exchange_declare(
            exchange=exchange_name, exchange_type=exchange_type, durable=True)
        for queue_name in queue_names:
            self.channel.queue_declare(queue=queue_name, durable=True)
            self.channel.queue_bind(exchange=exchange_name, queue=queue_name)


            print(self.channel.basic_publish(exchange=exchange_name,
                                  routing_key=queue_name,
                                  body=json.dumps(msg).encode("utf-8"),
                                  properties=pika.BasicProperties(delivery_mode=2)))
        #channel.close()

    def callback(self, channel, method, properties, body):
        print("[x] Received %r" % (body,))

    def read(self, exchange_name: str, exchange_type: str, queue_name: str):
        '''to consumer a msg
        '''
        #print(exchange_name, exchange_type, queue_name)
        #self.channel.exchange_declare(
        #    exchange=exchange_name, exchange_type=exchange_type, durable=True)
        self.channel.queue_declare(queue=queue_name, durable=True)
        #channel.queue_bind(exchange=exchange_name, queue=queue_name)

        self.channel.basic_consume(queue_name, self.callback, False)
        self.channel.start_consuming()

