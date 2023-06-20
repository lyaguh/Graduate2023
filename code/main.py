import os
import json
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    configs = json.load(open('./code/config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])


    data = DataLoader(
        split=configs['data']['train_test_split'],
        cols=configs['data']['columns']
    )

    # to load data from CSV files
    data.load_data_from_csv(
        filename=os.path.join('datasets', configs['data']['filename'])
    )


    timeframe = "M1"


    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'] #False
    )

    # x2, y2 = data.get_train_data2(
    #     seq_len=configs['data']['sequence_length'],
    #     normalise=configs['data']['normalise']  #False
    # )

    print("train data shapes: ", x.shape, y.shape)
    # print("train data shapes: ", x2.shape, y2.shape)
    #print(x, y);print(x2, y2); exit(1)


    model.train(
        x,
        y,
        epochs = configs['training']['epochs'],
        batch_size = configs['training']['batch_size'],
        save_dir = configs['model']['save_dir'],
        timeframe=timeframe
    )


    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'] #False
    )

    print("test data shapes: ", x_test.shape, y_test.shape)
    #print(x_test, y_test); exit(1)
    #print(x_test, y_test, x_test.shape, y_test.shape)

    model.eval_test(x_test,  y_test, verbose=2)

    _ev = model.eval_test2(x_test, y_test, verbose=0)
    print("### ", _ev, " ###")

    last_data_2_predict = data.get_last_data(-(configs['data']['sequence_length']-1), configs['data']['normalise'])
    print("*** ", -(configs['data']['sequence_length']-1), last_data_2_predict.size, "***")
    
    predictions2 = model.predict_point_by_point(last_data_2_predict)

    
    last_data_2_predict_prices = data.get_last_data(-(configs['data']['sequence_length']-1), False)
    last_data_2_predict_prices_1st_price = last_data_2_predict_prices[0][0]
    predicted_price = data.de_normalise_predicted(last_data_2_predict_prices_1st_price, predictions2[0])
    print("!!!!!", predictions2, predicted_price, "!!!!!")
    
    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    predictions = model.predict_point_by_point(x_test)
    # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^',predictions)
    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    plot_results(predictions, y_test)


if __name__ == '__main__':
    main()
