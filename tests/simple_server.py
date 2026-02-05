
import socket
import sys

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', 12345))
    server.listen(1)
    print("服务器启动在 127.0.0.1:12345")
    
    client, addr = server.accept()
    print(f"客户端连接: {addr}")
    
    # 接收消息
    data = client.recv(1024)
    print(f"收到: {data.decode('utf-8')}")
    
    # 发送回复
    client.send(b"Hello from Python Server!")
    
    client.close()
    server.close()

if __name__ == "__main__":
    main()
