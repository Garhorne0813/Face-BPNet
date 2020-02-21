from tkinter import *
import numpy as np
from PIL import Image, ImageTk


def sigmoid(net):
    return 1 / (1 + np.exp(-net))


# np.random.seed(1)

result = np.load('result.npz')
wp0 = result['arr_0']
wp1 = result['arr_1']
wp2 = result['arr_2']
b0 = result['arr_3']
b1 = result['arr_4']
b2 = result['arr_5']

test_data = np.load('test_data1.npy')
test_label = np.load('test_label.npy')
test_image = np.load('test_image.npy')

root = Tk()
root.geometry('300x200')
root.title('人脸识别')


def test():
    a = int(np.round(120 * np.random.random()))

    b = int(np.floor(a / 3) + 1)
    c = int(a % 3 + 8)
    s = 'att_faces/s' + str(b) + '/' + str(c) + '.pgm'
    im = PhotoImage(file=s)
    pic2.configure(image=im)
    pic2.image = im
    root.update()

    net0 = np.dot(test_data[a], wp0) + b0
    r0 = sigmoid(net0)

    net1 = np.dot(r0, wp1) + b1
    r1 = sigmoid(net1)

    net2 = np.dot(r1, wp2) + b2
    r2 = sigmoid(net2)

    x = r2.tolist()
    y = test_label[a].tolist()
    i1 = x.index(max(x))
    i2 = y.index(max(y))

    s = 'att_faces/s' + str(i1 + 1) + '/' + str(1) + '.pgm'
    im = PhotoImage(file=s)
    pic1.configure(image=im)
    pic1.image = im
    root.update()

    if i1 == i2:
        s = '正确'
    else:
        s = '错误'
    text.configure(text=s)


fm1 = Frame(root)
fm1.pack(side=TOP)

fm3 = Frame(fm1)
fm3.pack(side=LEFT)

fm4 = Frame(fm1)
fm4.pack(side=RIGHT)

pic1 = Label(fm3)
pic1.pack(side=TOP)

text_pic1 = Label(fm3, text='计算值').pack(side=BOTTOM)

pic2 = Label(fm4)
pic2.pack(side=TOP)

text_pic2 = Label(fm4, text='样本值').pack(side=BOTTOM)

fm2 = Frame(root)
fm2.pack(side=BOTTOM)

btn = Button(fm2, text='测试', command=test)
btn.pack()

text = Label(fm2, text=' ')
text.pack(side=BOTTOM)

root.mainloop()


