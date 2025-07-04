import datetime 

today = datetime.datetime.now() #현재 날짜와 시간정보를 준다 
print(today)

#파이썬의 dir함수가 있음 , 기본적으로 내부구조를 보여준다 
print( dir(datetime ) )

#날짜와 날짜의 연산을 수행하려면 날짜를 date클래스나 datetime클래스로 바꿔야 연산이 가능하다 
d1 = datetime.date(2025,7,31)
print(d1)

from datetime import date 
value = 1567345678 #타임스탬프
timestamp = date.fromtimestamp(value) #타임스탬프로부터 날짜를 얻어낸다 
print("date=", timestamp)


print( today.year )
print( today.month )
print( today.day )

print( today.hour )
print( today.minute )
print( today.second )

#요일 weekday()   0:월요일 1 2 3 4 5 6 
print(today.weekday())

from datetime import datetime, date, timedelta 
d1 = date(2025,7,4)
d2 = date(2025,12,31)
d3 = d2-d1 
print(d3) #올해말까지 남은 날짜 
print(type(d3)) #timedelta  타입이다 

#지금부터 3시간 뒤 
current = datetime.today() #현재 시간과 날짜 
after = current + timedelta(hours=3)
print("현재시간 :"  ,current)
print("3시간후 :"  ,after)

#오늘로부터 3일 뒤
current = datetime.today() #현재 시간과 날짜 
after = current + timedelta(days=3)
print("현재시간 :"  ,current)
print("3일후 :"  , after)

#오늘로부터 3일 하고 4시간 뒤
current = datetime.today() #현재 시간과 날짜 
after = current + timedelta(days=3, hours=4)
print("현재시간 :"  ,current)
print("3일후 :"  , after)

#오늘로부터 3일전 시간은 4시간후 
current = datetime.today() #현재 시간과 날짜 
after = current + timedelta(days=-3, hours=4)
print("현재시간 :"  ,current)
print("3일전 :"  , after)

#오늘로부터 2주전
current = datetime.today() #현재 시간과 날짜 
after = current + timedelta(weeks=2)
print("현재시간 :"  ,current)
print("2주전 :" , after)

print( dir(timedelta))

#날짜서식에 맞춰 출력하기 
print(today.strftime('%Y-%m-%d'))
print(today.strftime('%H:%M:%S'))
print(today.strftime('%Y-%m-%d %B %H:%M:%S '))

#타임존 - 각 나라별 시간 정보를 가져온다., 
import pytz 
format = "%Y-%m-%d %B %H:%M:%S"
local = datetime.now() 
print("현재지역시간 : ", local.strftime(format))

tz_NY = pytz.timezone("America/New_york") 
local = datetime.now(tz_NY) 
print("뉴욕현재시간 : ", local.strftime(format))

tz_NY = pytz.timezone("Europe/London") 
local = datetime.now(tz_NY) 
print("런던현재시간 : ", local.strftime(format))
