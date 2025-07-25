#직렬화 - 객체 자체를 파일이나 네트워크로 메모리 그대로 저장한다.
#역직렬화 - 파일이나 네트워크로부터 객체를 읽어들인다.

import pickle 
data = {'name':"홍길동", "age":23, 
            "phone":["010-0000-0001", "010-0000-0002"]}
#직렬화
with open("data.bin", "wb") as f:
    pickle.dump(data, f)

#역직렬화
with open("data.bin", "rb") as f:
    data2 = pickle.load(f)

print(data2)
