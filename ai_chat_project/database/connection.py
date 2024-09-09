import sqlite3
from sqlite3 import Connection
from threading import Lock
import queue
from datetime import datetime

class ConnectionPool:
    def __init__(self, database, max_connections):
        self.database = database
        self.max_connections = max_connections
        self.connections = queue.Queue(maxsize=max_connections)
        self.connection_count = 0
        self.lock = Lock()

    def get_connection(self) -> Connection:
        if not self.connections.empty():
            return self.connections.get()

        with self.lock:
            if self.connection_count < self.max_connections:
                connection = sqlite3.connect(self.database, 
                                             detect_types=sqlite3.PARSE_DECLTYPES,
                                             check_same_thread=False)
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('PRAGMA synchronous=NORMAL')
                self.connection_count += 1
                return connection

        return self.connections.get(block=True, timeout=30)

    def return_connection(self, connection: Connection):
        self.connections.put(connection)

    def close_all(self):
        while not self.connections.empty():
            connection = self.connections.get()
            connection.close()

DB_NAME = 'chatbot.db'
POOL_SIZE = 5
db_pool = ConnectionPool(DB_NAME, POOL_SIZE)

def init_db():
    conn = db_pool.get_connection()
    try:
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY, username TEXT UNIQUE)''')
        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (id INTEGER PRIMARY KEY, user_id INTEGER, conversation TEXT, timestamp TEXT, context TEXT, title TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS chat_state
                     (id INTEGER PRIMARY KEY, user_id INTEGER, conversation TEXT, document_content TEXT, context TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS sessions
                     (id TEXT PRIMARY KEY, user_id INTEGER, expiry TIMESTAMP)''')

        conn.commit()
    finally:
        db_pool.return_connection(conn)

def close_db_connections():
    db_pool.close_all()