#插入排序 向有序的数组插入一个数，使之插入到对应的位置 并移动其它的元素
arr = [3,1,6,5,3,1,243,5,326,9]

def insert_sort(arr):
    #将第一个元素看为有序的 直接从第二个元素开始插入
    for i in range(1,len(arr)):
        tmp = arr[i]
        for j in range(0,i+1):
            #判断是否循环到了取出的那个数的索引位置 如果是就将临时数添到这个位置 这个数一定是比前边的所有位置的数大
            if j == i:
                arr[i]=tmp
            #判断 取出的数是不是比循环到的这个数大 如果是就往后找比他小的数并顶替他成为这个位置上的数
            if(tmp >= arr[j]):
                continue
            #将新的数插入到j的位置并移动后边的数索引+1
            arr[j],tmp = tmp,arr[j]
insert_sort(arr)
print(arr)

