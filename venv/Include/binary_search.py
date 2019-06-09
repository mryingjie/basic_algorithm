
arr = [0,1,2,3,4,5,6,7]

def division(arr,head,tail,n):
    if head > tail:
        return -1
    mid = (head + tail) // 2
    if arr[mid] == n:
        return mid
    if arr[mid] > n:
        return division(arr,head,mid-1,n)
    if arr[mid] < n:
        return division(arr,mid+1,tail,n)

# 二分查找法 递归
def binary_search_recursive(arr,n):
    head = 0
    tail = len(arr) -1
    return division(arr,head,tail,n)

# 二分查找法 循环
def binary_search_cycle(arr,n):
    head = 0
    tail = len(arr) -1
    while head <= tail:
        mid = (head + tail) // 2
        if arr[mid] == n:
            index = mid
            return mid;
        elif arr[mid] > n:
            tail = mid - 1
        else:
            head = mid + 1
    return -1



print(binary_search_recursive(arr,2))
print(binary_search_cycle(arr,7))