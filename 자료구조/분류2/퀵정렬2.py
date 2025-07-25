def write(arr):
    for num in arr:
        print(f"{num:4}", end='')
    print()

def quicksort(arr, start, end):
    if start >= end:
        return

    pivot = arr[start]
    left = start + 1
    right = end

    while left <= right:
        while left <= end and arr[left] < pivot:
            left += 1
        while right > start and arr[right] > pivot:
            right -= 1

        if left < right:
            arr[left], arr[right] = arr[right], arr[left]
        else:
            break

    arr[start], arr[right] = arr[right], arr[start]
    write(arr)  # 중간 상태 출력

    quicksort(arr, start, right - 1)
    quicksort(arr, right + 1, end)

# 실행
arr = [5, 1, 6, 4, 8, 3, 7, 9, 2, 10]
quicksort(arr, 0, len(arr) - 1)

print("최종 정렬 결과:")
write(arr)
