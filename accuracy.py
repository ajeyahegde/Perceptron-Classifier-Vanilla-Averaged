file1 = open("percepoutput.txt", "r", encoding='utf8')
file2 = open("dev-key.txt","r",encoding='utf8')

correct1 = 0
correct2 = 0

total = 0
for line1 in file1:
    arr1 = line1.strip().split(" ")
    arr2 = file2.readline().strip().split(" ")
    if arr1[1] == arr2[1]:
        correct1 += 1
    if arr1[2] == arr2[2]:
        correct2 += 1
    total += 1

print(correct1)
print(correct2)

accuracy1 = correct1/total
accuracy2 = correct2/total

accuracy = (correct1+correct2)/(2*total)
print(accuracy)
print(accuracy1)
print(accuracy2)
