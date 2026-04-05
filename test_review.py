"""Sample file to test PR Review AI end-to-end."""

import os
import json


def read_config(path):
    f = open(path, "r")
    data = json.loads(f.read())
    return data


def process_users(db_conn, username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    result = db_conn.execute(query)
    return result


def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    avg = total / len(numbers)
    return avg
