import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_type = np.dtype([("word", 'S30'), ("count", np.longlong)])
    data = np.loadtxt("google-books-common-words.txt", dtype=data_type, delimiter="\t")
    data = data[0:1000]

    log_count = np.vstack((list(map(lambda x: np.log(x), list(range(1, len(data) + 1)))), list(map(lambda x: np.log(x), data["count"]))))
    model = LinearRegression()
    x_train = np.array(log_count[0]).reshape(-1, 1)
    y_train = np.array(log_count[1]).reshape(-1, 1)
    model.fit(x_train, y_train)
    print("-alpha is ", model.coef_)
    print("c is ", model.intercept_)
    y2 = list(map(lambda x: np.exp(model.intercept_[0] + model.coef_[0][0] * x), x_train))
    origin, = plt.loglog(list(range(len(data))), data["count"])
    regression, = plt.loglog(list(range(len(data))), y2)
    # plt.plot(list(range(len(data))), data["count"])
    # plt.plot(list(range(len(data))), log_count[1])
    plt.legend([origin, regression], ["original data", "regression"], loc=1)
    plt.xlabel('frequency rank')
    plt.ylabel('count')
    plt.show()
