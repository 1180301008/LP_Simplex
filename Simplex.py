import numpy as np
"""
A = (p1 p2 ... pn)

cB  xB  b   x1  x2  ...  xn theta
        
        B-1b B-1A
    -z  xi-ciB-1pi
"""

class Simplex:
    # 输入必须已经转化位标准型
    def __init__(self,A,b,c,xB):
        self.A = A.T # A[i]为列向量
        self.b = b
        self.c = c
        self.xB = xB
        self.m,self.n = A.shape

    # 根据初始基初始单纯形表
    def create_table(self):
        self.B = self.A[self.xB]
        self.cB = self.c[self.xB]
        self.sigma = self.c - np.inner(self.cB,self.A) # 检验数
        self.theta = np.zeros(self.m)

    # 最优性检验
    def is_best(self):
        flag = 0
        if (self.sigma <= 0).all():
            flag = 1
            for i in range(self.n):
                if i not in self.xB and self.sigma[i] == 0: # 无穷多最优解
                    flag = 2
                    break
        else:
            for i in range(self.n):
                if i not in self.xB and self.sigma[i] > 0 and (self.A[i] <= 0).all():
                    flag = 3
        return flag

    def show_table(self):
        print('当前基变量', list(map(lambda n:n+1, self.xB)))
        print('----------------------------------')
        tmp = ''
        for i in range(self.n):
            tmp = tmp + 'x' + str(i+1) +'\t'*2
        print('cB\t\txB\t\tb\t\t'+tmp)
        for i in range(self.m):
            tmp = ''
            tmp += "%.2f" % (self.cB[i]) + '\t' + "%.2f" % (self.xB[i]+1)+ '\t' + "%.2f" % (self.b[i])+'\t'
            for j in range(self.n):
                tmp += "%.2f" % (self.A[j,i]) + '\t'
            print(tmp)
        tmp = '\t'*2 + '-z' + '\t' + "%.2f" % (-self.get_Z()) + '\t'*2
        for i in range(self.n):
            tmp += "%.2f" % (self.sigma[i]) + '\t'
        print(tmp)
        print('----------------------------------')

    # 选择入基、出基，重新计算单纯表
    def update(self):
        # 选择入基
        self.inVar = np.where(self.sigma == max(self.sigma))[0][0] # 最大的那
        print('入基变量: x',self.inVar+1)
        pi = self.A[self.inVar]
        # 计算theta
        tmp = 'theta:\t'
        for i in range(self.m):
            if(pi[i] <= 0):
                self.theta[i] = float('inf')
            else:
                self.theta[i] = self.b[i] / pi[i]
            tmp += str(self.theta[i])+'\t'*2
        print(tmp)
        # 选择出基
        self.outVar = np.where(self.theta == min(self.theta))[0][0]
        print('出基变量: x',self.xB[self.outVar])

        # 更新表
        self.xB[self.outVar] = self.inVar
        self.cB = self.c[self.xB]
        self.B = self.A[self.xB]
        B_inv = np.linalg.inv(self.B).T # B-1
        self.A = np.dot(B_inv,self.A.T).T
        self.b = np.dot(B_inv,self.b)
        self.sigma = self.c - np.inner(self.cB, self.A)  # 检验数


    # 获得z
    def get_Z(self):
        return np.dot(self.cB,self.b)

    def process(self):
        self.create_table()
        self.show_table()
        while(True):
            if self.is_best() == 0:
                self.update()
                self.show_table()
            elif self.is_best() == 1:
                print('此问题有唯一最优解')
                print('z=', self.get_Z())
                break
            elif self.is_best() == 2:
                print('无穷多最优解')
                print('z=',self.get_Z())
                break
            else:
                print('无最优解')
                break


def main():
    # A = np.matrix([[1, 4, 2, 1, 0],
    #                [1, 2, 4, 0, 1]])
    # b = np.matrix([[48, 60]])
    # c = np.matrix([[6, 14, 13, 0, 0]])
    base = list(map(lambda n:n-1, [4,5]))
    xB = np.array(base)

    A = np.array([[1, 4, 2, 1, 0],
                   [1, 2, 4, 0, 1]])
    b = np.array([48,60])
    c = np.array([6, 14, 13, 0, 0])

    lp = Simplex(A,b,c,xB)
    lp.process()



if __name__ == '__main__':
    main()