arr = [32, 23, 32, 4,4,54,2,3,3]


def merge_sort(arr):
    tmp = [0] * (len(arr))
    merge_sort_base(arr, 0, len(arr) - 1, tmp)


# 递归分 直到只剩下一个元素
def merge_sort_base(arr, l, r, tmp):
    if l >= r:
        return
    mid = (l + r) // 2
    merge_sort_base(arr, l, mid, tmp)
    merge_sort_base(arr, mid + 1, r, tmp)
    # 合并
    merage(arr, l, mid, r, tmp)


def merage(arr, l, m, r, tmp):
    tmp_l = l  # 左序指针
    tmp_r = m + 1  # 右序指针
    tmp_i = 0
    # 判断大小并放入原始数组
    while tmp_l <= m and tmp_r <= r:
        if arr[tmp_l] <= arr[tmp_r]:
            tmp[tmp_i] = arr[tmp_l]
            tmp_l += 1
        else:
            tmp[tmp_i] = arr[tmp_r]
            tmp_r += 1
        tmp_i += 1

    # 将左边剩余的元素填入原始数组
    while tmp_l <= m:
        tmp[tmp_i] = arr[tmp_l]
        tmp_l += 1
        tmp_i += 1

    while tmp_r <= r:
        tmp[tmp_i] = arr[tmp_r]
        tmp_r += 1
        tmp_i += 1

    tmp_i = 0
    # 将tmp中的元素放入数组中
    while l <= r:
        arr[l] = tmp[tmp_i]
        tmp_i += 1
        l += 1


merge_sort(arr)
print("排序完成")
print(arr)
