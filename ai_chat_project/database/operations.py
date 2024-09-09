import sqlite3
import streamlit as st
from .connection import db_pool

def safe_db_operation(operation):
    conn = db_pool.get_connection()
    try:
        with conn:
            operation(conn.cursor())
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        conn.rollback()
    finally:
        db_pool.return_connection(conn)

def execute_query(query, params=None):
    conn = db_pool.get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
    finally:
        db_pool.return_connection(conn)

def execute_insert(query, params):
    conn = db_pool.get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid
    finally:
        db_pool.return_connection(conn)