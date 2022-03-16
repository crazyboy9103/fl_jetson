* FL framework
    * (TF의 경우) 서버 열기
    ```python3 multi_server.py --exp (실험번호 1,2,3,4 중 하나) --num (각 클라이언트가 학습할 데이터 샘플 개수) --cli (전체 클라이언트 수) --round (전체 학습할 라운드) --data (데이터 mnist, cifar10, cifar100, fmnist 중 하나) --host (호스트 주소 default=127.0.0.1) --port (호스트 포트 번호) --epochs (각 클라이언트가 학습할 에폭 수) --batch (배치 크기)```

    e.g. ```python3 multi_server.py --exp 1 --num 200 --cli 5 --round 5 --data mnist --host 127.0.0.1 --port 20000 --epochs 5 --batch 8```

    * 클라이언트 테스트
    ```python3 test_multi_clients.py --host (호스트 주소)--port (호스트 포트) --cli (전체 클라이언트 수)```
    
    e.g. ```python3 test_multi_clients.py --host 127.0.0.1 --port 20000 --cli 5```


* pickle_socket.py : pickle로 serialize된 객체를 TCP를 통해 주고받는 라이브러리
    * 사용법은 test_server.py와 test_client.py 참고