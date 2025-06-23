class User:
    count = 0

    def __init__(self, username):
        self.username = username
        User.count += 1

    @classmethod
    def from_email(cls, email):
        """이메일에서 사용자 이름만 추출하여 객체 생성"""
        username = email.split('@')[0]
        return cls(username)

    @classmethod
    def get_user_count(cls):
        """생성된 사용자 수 반환"""
        return cls.count


# 사용 예시
print("=== classmethod 예시 ===")
user1 = User("alice")
user2 = User.from_email("bob@example.com")  # 이메일로부터 객체 생성
print("사용자 수:", User.get_user_count())
print("user2 이름:", user2.username)
