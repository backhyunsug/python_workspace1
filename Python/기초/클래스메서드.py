class Book:
    total_books = 0  # 클래스 변수

    def __init__(self, title, price):
        self.title = title
        self.price = price
        Book.total_books += 1

    @staticmethod
    def calculate_discount_price(price, rate):
        """정적 메서드: 단순 계산 로직, 클래스/인스턴스와 무관"""
        return price * (1 - rate)

    @classmethod
    def get_total_books(cls):
        """클래스 메서드: 클래스 변수 접근"""
        return cls.total_books


# 인스턴스 생성
book1 = Book("Python Basics", 15000)
book2 = Book("Advanced Python", 20000)

# 정적 메서드 사용 (할인 가격 계산)
discounted = Book.calculate_discount_price(20000, 0.1)
print(f"할인가: {discounted}원")  # 할인가: 18000.0원

# 클래스 메서드 사용 (책 수 세기)
print(f"총 책 수: {Book.get_total_books()}권")  # 총 책 수: 2권
