def write(a):
    for num in a:
        print(f"{num:4}", end="")
    print()

def quicksort(a, start, end):
    if start >= end:
        return

    left = start + 1
    right = end

    while left <= right:
        while left <= end and a[left] < a[start]:
            left += 1
        while right > start and a[right] > a[start]:
            right -= 1

        if left < right:
            a[left], a[right] = a[right], a[left]
        else:
            break

    # swap pivot with a[right]
    a[start], a[right] = a[right], a[start]

    write(a)

    quicksort(a, start, right - 1)
    quicksort(a, right + 1, end)

if __name__ == "__main__":
    a = [5, 1, 6, 4, 8, 3, 7, 9, 2, 10]
    quicksort(a, 0, len(a) - 1)
