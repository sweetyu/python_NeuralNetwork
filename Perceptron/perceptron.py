import numpy as np
import sys

# parameter init
inputItem = np.array([])
targetItem = np.array([])
w = np.array([1,1,1])
b = 0.5
hardlim = lambda x: 1 if x >= 0 else 0

# update parameters 
def update(item, e):
    global w, b
    w += e * item
    b += e
    print "inputItem=",item,"e=",e,"new_w=",w,"new_b=",b

# check whether the classification is correct or not
def check(item, target):
    global w, b
    p = item.transpose()
    a = hardlim(np.dot(w, p) + b)
    e = target - a
    return e  #e/10   #e/20

# calculate the parameters to the model
def cal():
    ndim = inputItem.ndim
    iter_count = 0
    count = 0    
    while True:
        for i in range(ndim):
            if(count>=ndim):           
                return iter_count
            e = check(inputItem[i], targetItem[i])
            iter_count += 1
            if(e != 0):
                update(inputItem[i], e)
                count = 0
            else:
                count += 1
            print "QualifiedItemNum:",count,"IterCount:",iter_count

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: python preceptron.py train.txt output.txt model.txt"
        exit(0)

    inputItem = np.loadtxt(sys.argv[1])
    targetItem = np.loadtxt(sys.argv[2])
    modelFile = file(sys.argv[3], 'w')
    #set w to zero matrix
    w = np.zeros([1, inputItem.shape[1]])

    # train start
    iter_count = cal()

    modelFile.write(" ".join(str(i) for i in w) + "\n")
    modelFile.write(str(b))
    modelFile.close()

    print "After %d iterations, the parameters converge to:",iter_count
    print "W: ", w
    print "b: ", b  
