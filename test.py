class Solution:
    def dailyTemperatures(self, ops):
        op, index = [], -1
        for ch in ops:
            if len(ch) > 1 or ('0' <= ch <= '9'):
                op.append(int(ch))
                index += 1
            elif ch == 'C':
                del(op[index])
                index -= 1
            elif ch == 'D':
                op.append(op[index] * 2)
                index += 1
            elif ch == '+':
                op.append(op[index] + op[index-1])
                index += 1
        return sum(op)

ops = ["1","C","-62","-45","-68"]
print(Solution().dailyTemperatures(ops))