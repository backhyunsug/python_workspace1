"""
문제: 공주님의 정원 (백준 2457)
📌 문제 설명 요약

매일 꽃 한 송이는 피어 있고, 공주님은 3월 1일부터 11월 30일까지 매일 꽃이 피어있기를 원합니다.

각 꽃은 피는 날과 지는 날이 정해져 있습니다.
(예: 3월 1일 → 301, 11월 30일 → 1130)

가장 적은 수의 꽃을 선택해서  : 최대한 오래 피어있는 꽃을 고르자
3월 1일부터 11월 30일까지 끊기지 않게 덮도록 하세요

5
1 1 5 31
1 1 6 10              101    610이 선택됨     1 
5 15 8 15
6 10 12 10
8 15 11 30

날짜를 MMDD → 정수형으로 변환한다.

피는 날 기준으로 정렬하되, 같은 날 피는 꽃은 늦게 지는 것부터 정렬

시작 날짜를 301로 두고,

매번 현재 날짜 이전에 피었고 가장 늦게 지는 꽃을 선택

날짜를 그 꽃의 지는 날로 갱신

이를 반복

도중에 다음 꽃이 없으면 종료 (불가능)
"""

def date_to_int(month, day):
    return month * 100 + day

def solve_flowers(N, flowers):
    flowers = [(date_to_int(s_m, s_d), date_to_int(e_m, e_d)) for s_m, s_d, e_m, e_d in flowers]
    print(flowers)
    flowers.sort()  # 피는 날짜 기준 정렬 (같으면 지는 날짜 자동 정렬)

    start = 301
    end = 1130
    i = 0
    count = 0
    max_end = 0

    while start <= end:  #꽃이 지는 날보다 피는날이 빠르다
        updated = False
        #301을 포함하는 요소 중에 끝나는 날이 가장 큰 날을 고른다 
        while i < N and flowers[i][0] <= start: #이미 피어있는걸 고른다
            if flowers[i][1] > max_end: #꽃이 지는 날중에 가장 큰걸 고른다 
                max_end = flowers[i][1]  #531 
                updated = True
            i += 1

        print(i, max_end)

        if not updated:#조건을 만족하는게하나도 없었다
            return 0  # 덮을 수 없음
        start = max_end #start를 610 으로 옮기고 또 작업 
        count += 1

    return count


N = 5
flowers = [
    (1, 1, 5, 31),
    (1, 1, 6, 10),
    (5, 15, 8, 15),
    (6, 10, 12, 10),
    (8, 15, 11, 30)
]

print(solve_flowers(N, flowers))  