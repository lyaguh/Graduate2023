import json
import os
from flask import Flask, render_template
from core.data_processor import DataLoader
from core.model import Model
app = Flask(__name__)

configs = json.load(open('./code/config.json', 'r'))

data = DataLoader(
    split=configs['data']['train_test_split'],
    cols=configs['data']['columns']
)

data_true = DataLoader(
    split=configs['data']['train_test_split'],
    cols=configs['data']['columns']
)

data_week = DataLoader(
    split=configs['data']['train_test_split'],
    cols=configs['data']['columns']
)

data_true_week = DataLoader(
    split=configs['data']['train_test_split'],
    cols=configs['data']['columns']
)

# to load data from CSV files
data.load_data_from_csv(
    filename=os.path.join('datasets', configs['data']['filename'])
)

data_true.load_data_from_csv(
    filename=os.path.join('datasets', 'USD_RUB.csv')
)

data_true_week.load_data_from_csv(
    filename=os.path.join('datasets', 'USD_RUB_W1.csv')
)

data_week.load_data_from_csv(
    filename=os.path.join('datasets', 'USD_RUB_W.csv')
)

x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'] #False
    )

model_day = Model()
model_week = Model()
model_day.load_model('saved_models/19062023-132338-e10_D1.h5')
model_week.load_model('saved_models/19062023-125414-e10_W1.h5')

# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
def predict(data,model):
    last_data_2_predict = data.get_last_data(-(configs['data']['sequence_length']-1), configs['data']['normalise'])
    predictions2 = model.predict_point_by_point(last_data_2_predict)
    last_data_2_predict_prices = data.get_last_data(-(configs['data']['sequence_length']-1), False)
    last_data_2_predict_prices_1st_price = last_data_2_predict_prices[0][0]
    predicted_price = data.de_normalise_predicted(last_data_2_predict_prices_1st_price, predictions2[0])
    predictions = model.predict_point_by_point(x_test)
    return [predicted_price,predictions]

predicted_day = predict(data,model_day)
predicted_price_day, predictions_day = predicted_day[0],predicted_day[1]

predicted_week = predict(data_week,model_week)
predicted_price_week= predicted_week[0]

@app.route("/")
def index():
     
                                               
    data_y_test,data_predictions,labels = [],[],[]
    for i in y_test:
        data_y_test.append(i[0])
    for i in predictions_day:
        data_predictions.append(i)
    for i in range(0,len(data_y_test)):
        labels.append(i)

    return render_template('index.html',true_price=str(data_true.get_last_price()),true_price_week=str(data_true_week.get_last_price()),predicted_price_day=str(predicted_price_day),predicted_price_week=str(predicted_price_week),labels=labels,data1=data_y_test,data2=data_predictions)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)