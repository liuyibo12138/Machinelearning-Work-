import csv  #导入csv模块
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


# 代价函数
def compute_cost(X, w, b, y):
    m = X.shape[0]
    cost_all = 0
    f_wb = w * X + b
    cost_all += (f_wb - y) ** 2
    cost_all = np.sum(cost_all)
    return cost_all / (2 * m)


# 计算梯度
def compute_gradient(X, w, b, y):
    m = X.shape[0]
    dJ_dw = 0
    dJ_db = 0
    f_wb = w * X + b
    dJ_dw += (f_wb - y) * X
    dJ_db += (f_wb - y)
    dJ_dw = np.sum(dJ_dw)
    dJ_db = np.sum(dJ_db)
    dJ_dw = dJ_dw / m
    dJ_db = dJ_db / m
    return dJ_dw, dJ_db


# 梯度下降
def gradient_decent(X, w_in, b_in, y, alpha, iterate):
    w = w_in
    b = b_in
    for i in range(1, iterate):
        dJ_dw, dJ_db = compute_gradient(X, w, b, y)
        w = w - alpha * dJ_dw
        b = b - alpha * dJ_db
        if i % 100 == 0:
            print(str(i) + "/" + str(iterate) + "  Cost:" + str(compute_cost(X, w, b, y)))
    return w, b

#归一化
def normalization(data):
    max_value = max(data)  # 最大值
    min_value = min(data)  # 最小值
    norm = []  # 返回的归一化后的数组
    for x in data:
        norm.append((x - min_value) / (max_value - min_value))  # 归一化计算公式
    return norm

with open('D:/py2.7/机器学习课程设计/used_car_train_20200313.csv') as csv_file:  #打开CSV文件
    csv_reader = csv.reader(csv_file, delimiter=',')  #使用csv.reader读取文件内容，设置delimiter分割符
    line_count = 0  #声明一个行计数器
    line_counts=[]
    names=[]
    prices=[]
    regdate=[]
    models=[]
    brands=[]
    bodytype=[]
    fueltypes=[]
    gearboxs=[]
    powers=[]
    kilometers=[]
    notRepairedDamages=[]
    regioncodes=[]
    seller=[]
    offertype=[]
    createdates=[]
    v0=[]
    v1=[]
    v2 = []
    v3 = []
    v4= []
    v5 = []
    v6 = []
    v7= []
    v8 = []
    v9 = []
    v10 = []
    v11 = []
    v12 = []
    v13 = []
    v14 = []

    for row in csv_reader:  #循环读取csv文件数据
        row = ", ".join(row).split()
        if line_count == 0:  #第一行为标题
            print(f'Columns: {", ".join(row)}')  #打印csv文件的列名


        names.append(row[1])
        regdate.append(row[2])
        models.append(row[3])
        brands.append(row[4])
        bodytype.append(row[5])
        fueltypes.append(row[6])
        gearboxs.append(row[7])
        powers.append(row[8])
        kilometers.append(row[9])
        notRepairedDamages.append(row[10])
        regioncodes.append(row[11])

        seller.append(row[12])
        offertype.append(row[13])
        createdates.append(row[14])



        prices.append(row[15])
        line_counts.append(line_count)
        line_count += 1  #行计数器累加
    print(f'Processed {line_count} ')  #打印总行数





    powers, priceofpowers = zip(*[(p, c) for p, c in zip(powers, prices) if p!='-'])
    powers = np.array(list(map(float, powers[1:])))
    priceofpowers = np.array(list(priceofpowers[1:]))


    line_counts = line_counts[1:]
    names = np.array(list(map(float, names[1:])))
    prices = np.array(list(map(float, prices[1:])))
    regdate = np.array(list(map(float, regdate[1:])))
    models = np.array(list(map(float, models[1:])))
    brands = np.array(list(map(float, brands[1:])))
    bodytype =  np.array(list(map(float, bodytype[1:])))
    fueltypes = np.array(list(map(float, fueltypes[1:])))

    gearboxs, priceofgearboxs = zip(*[(p, c) for p, c in zip(gearboxs, prices) if p != '-'])
    gearboxs = np.array(list(map(float, gearboxs[1:])))
    priceofgearboxs = np.array(list(priceofgearboxs[1:]))



    kilometers, priceofkilometers = zip(*[(p, c) for p, c in zip(kilometers, prices) if p != '-'])
    kilometers = np.array(list(map(float, kilometers[1:])))
    priceofkilometers = np.array(list(priceofkilometers[1:]))

    notRepairedDamages, priceofnotRepairedDamages = zip(*[(p, c) for p, c in zip(notRepairedDamages, prices) if p!='-'])
    priceofnotRepairedDamages= np.array(list(priceofnotRepairedDamages[1:]))
    notRepairedDamages = np.array(list(map(float, notRepairedDamages[1:])))
    print(notRepairedDamages)
    regioncodes = np.array(list(map(float, regioncodes[1:])))
    seller = np.array(list(map(float, seller[1:])))
    offertype = np.array(list(map(float, offertype[1:])))
    createdates = np.array(list(map(float, createdates[1:])))

    #归一化
    #names = normalization(names)
    #prices= normalization(prices)
    #models= normalization(models)
    #print(f'names:{names}')
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(30,14))
    plt.subplots_adjust(hspace=1,wspace=0.4)
    ax[0][0].scatter(names, prices)
    ax[0][0].set_xlabel('name')
    ax[0][0].set_ylabel('price')
    ax[0][0].set_title('name')

    ax[0][1].scatter(regdate, prices)
    ax[0][1].set_xlabel('regdate')
    ax[0][1].set_ylabel('price')
    ax[0][1].set_title('regdate')

    ax[0][2].scatter(models, prices)
    ax[0][2].set_xlabel('models')
    ax[0][2].set_ylabel('price')
    ax[0][2].set_title('models')

    ax[0][3].scatter(brands, prices)
    ax[0][3].set_xlabel('brands')
    ax[0][3].set_ylabel('price')
    ax[0][3].set_title('brands')

    ax[1][0].scatter(bodytype, prices)
    ax[1][0].set_xlabel('bodytype')
    ax[1][0].set_ylabel('price')
    ax[1][0].set_title('bodytype')

    ax[1][1].scatter(fueltypes, prices)
    ax[1][1].set_xlabel('fueltypes')
    ax[1][1].set_ylabel('price')
    ax[1][1].set_title('fueltypes')

    ax[1][2].scatter(gearboxs, priceofgearboxs)
    ax[1][2].set_xlabel('gearboxs')
    ax[1][2].set_ylabel('price')
    ax[1][2].set_title('gearboxs')


    print(priceofpowers)
    ax[1][3].yaxis.set_major_locator(MultipleLocator(50000))
    ax[1][3].scatter(powers, priceofpowers)
    ax[1][3].set_xlabel('powers')
    ax[1][3].set_ylabel('price')
    ax[1][3].set_title('powers')

    ax[2][0].scatter(kilometers, priceofkilometers)
    ax[2][0].set_xlabel('kilometer')
    ax[2][0].set_ylabel('price')
    ax[2][0].set_title('kilometer')

    ax[2][1].scatter(notRepairedDamages, priceofnotRepairedDamages)
    ax[2][1].set_xlabel('notRepairedDamage')
    ax[2][1].set_ylabel('price')
    ax[2][1].set_title('notRepairedDamage')

    ax[2][2].scatter(regioncodes, prices)
    ax[2][2].set_xlabel('regionCode')
    ax[2][2].set_ylabel('price')
    ax[2][2].set_title('regionCode')

    ax[2][3].scatter(seller, prices)
    ax[2][3].set_xlabel('seller')
    ax[2][3].set_ylabel('price')
    ax[2][3].set_title('seller')

    ax[3][0].scatter(offertype, prices)
    ax[3][0].set_xlabel('offerType')
    ax[3][0].set_ylabel('price')
    ax[3][0].set_title('offerType')

    ax[3][1].scatter(createdates, prices)
    ax[3][1].set_xlabel('creatDate')
    ax[3][1].set_ylabel('price')
    ax[3][1].set_title('creatDate')






    # w = 0.01
    # b = 0
    # X = np.arange(0, 5, 0.1)
    # axs.plot(X, w * X + b, label="init")
    plt.show()
    # X_in = np.array(line_counts)
    # y_in = np.array(names)
    #
    # fin_w, fin_b = gradient_decent(X_in, w, b, y_in, 1.56e-3, 10000)
    # print(fin_w, fin_b)
    # axs.plot(X, fin_w * X + fin_b, label="Pridict")
    # plt.legend()
    # plt.show()



