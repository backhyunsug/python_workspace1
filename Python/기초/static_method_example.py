class TemperatureConverter:
    """섭씨 <-> 화씨 변환: 클래스나 인스턴스 상태와 무관"""

    @staticmethod
    def celsius_to_fahrenheit(celsius):
        return celsius * 9/5 + 32

    @staticmethod
    def fahrenheit_to_celsius(fahrenheit):
        return (fahrenheit - 32) * 5/9


# 사용 예시
print("=== staticmethod 예시 ===")
print("섭씨 25도 -> 화씨:", TemperatureConverter.celsius_to_fahrenheit(25))
print("화씨 77도 -> 섭씨:", TemperatureConverter.fahrenheit_to_celsius(77))
