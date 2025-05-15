# backend.py
from flask import Flask, request
import socket

app = Flask(__name__)

@app.route('/')
def hello_user():
    user_id = request.args.get("user_id", "Guest")
    instance = socket.gethostname()
    
    return f"Hello, {user_id}! This response is from instance: {instance}"

if __name__ == '__main__':
    # TODO: Change the port if needed (default is 5001)
    app.run(host='0.0.0.0', port=5001)