from flask import Flask, render_template, send_file  # request, redirect
from predict import predict, getLatest, getMonthdays
from portfolio import getTPA
from exporter import save_to_file
# import tensorflow as tf


app = Flask("cocoinweb")

# model_btc = tf.keras.models.load_model(
#     '10_latest_btc_30d_lb7_loss_0.0395_val_loss_0.0422.hdf5')

# model_eth = tf.keras.models.load_model(
#     '1_eth_30d_lb7_loss_0.0268_val_loss_0.0219.hdf5')


@app.route("/")
def minjicoin():
    term_btc, predicted_btc, actual_btc = getTPA("KRW-BTC")  # , model_btc

    term1_btc = term_btc[0]
    term2_btc = term_btc[1]
    term3_btc = term_btc[2]
    term4_btc = term_btc[3]
    term5_btc = term_btc[4]

    predicted1_btc = predicted_btc[0]
    predicted2_btc = predicted_btc[1]
    predicted3_btc = predicted_btc[2]
    predicted4_btc = predicted_btc[3]
    predicted5_btc = predicted_btc[4]

    actual1_btc = actual_btc[0]
    actual2_btc = actual_btc[1]
    actual3_btc = actual_btc[2]
    actual4_btc = actual_btc[3]
    actual5_btc = actual_btc[4]

    term_eth, predicted_eth, actual_eth = getTPA("KRW-ETH")  # , model_eth

    term1_eth = term_eth[0]
    term2_eth = term_eth[1]
    term3_eth = term_eth[2]
    term4_eth = term_eth[3]
    term5_eth = term_eth[4]

    predicted1_eth = predicted_eth[0]
    predicted2_eth = predicted_eth[1]
    predicted3_eth = predicted_eth[2]
    predicted4_eth = predicted_eth[3]
    predicted5_eth = predicted_eth[4]

    actual1_eth = actual_eth[0]
    actual2_eth = actual_eth[1]
    actual3_eth = actual_eth[2]
    actual4_eth = actual_eth[3]
    actual5_eth = actual_eth[4]
    return render_template('home.html', **locals())


@app.route("/BTCprediction")
def predictBTC():
    # symbol = request.args.get('symbol')
    # if symbol:
    #     symbol = symbol.upper()
    # else:
    #     return redirect("/")
    # ticker = f"KRW-{symbol}"
    # preds_inversed = predict(ticker)
    preds_inversed = predict("KRW-BTC")
    y_preds = preds_inversed.tolist()
    days = getMonthdays()
    price_latest = getLatest("KRW-BTC")
    latest = []
    for i in range(30):
        latest.append(price_latest)
    # day01 = y_preds[0]
    # day02 = y_preds[1]
    # day03 = y_preds[2]
    # day04 = y_preds[3]
    # day05 = y_preds[4]
    # day06 = y_preds[5]
    # day07 = y_preds[6]
    # day08 = y_preds[7]
    # day09 = y_preds[8]
    # day10 = y_preds[9]
    # day11 = y_preds[10]
    # day12 = y_preds[11]
    # day13 = y_preds[12]
    # day14 = y_preds[13]
    # day15 = y_preds[14]
    # day16 = y_preds[15]
    # day17 = y_preds[16]
    # day18 = y_preds[17]
    # day19 = y_preds[18]
    # day20 = y_preds[19]
    # day21 = y_preds[20]
    # day22 = y_preds[21]
    # day23 = y_preds[22]
    # day24 = y_preds[23]
    # day25 = y_preds[24]
    # day26 = y_preds[25]
    # day27 = y_preds[26]
    # day28 = y_preds[27]
    # day29 = y_preds[28]
    # day30 = y_preds[29]
    return render_template('btc_prediction.html', **locals())
# symbol=symbol, days=days, y_preds=y_preds, latest=latest


@app.route("/ETHprediction")
def predictETH():
    preds_inversed = predict("KRW-ETH")
    y_preds = preds_inversed.tolist()
    days = getMonthdays()
    price_latest = getLatest("KRW-ETH")
    latest = []
    for i in range(30):
        latest.append(price_latest)
    # day01 = y_preds[0]
    # day02 = y_preds[1]
    # day03 = y_preds[2]
    # day04 = y_preds[3]
    # day05 = y_preds[4]
    # day06 = y_preds[5]
    # day07 = y_preds[6]
    # day08 = y_preds[7]
    # day09 = y_preds[8]
    # day10 = y_preds[9]
    # day11 = y_preds[10]
    # day12 = y_preds[11]
    # day13 = y_preds[12]
    # day14 = y_preds[13]
    # day15 = y_preds[14]
    # day16 = y_preds[15]
    # day17 = y_preds[16]
    # day18 = y_preds[17]
    # day19 = y_preds[18]
    # day20 = y_preds[19]
    # day21 = y_preds[20]
    # day22 = y_preds[21]
    # day23 = y_preds[22]
    # day24 = y_preds[23]
    # day25 = y_preds[24]
    # day26 = y_preds[25]
    # day27 = y_preds[26]
    # day28 = y_preds[27]
    # day29 = y_preds[28]
    # day30 = y_preds[29]
    return render_template('eth_prediction.html', **locals())


@app.route("/BTCexport")
def exportBTC():
    filename = "btcMonthPrediction.csv"
    preds_inversed = predict("KRW-BTC")
    y_preds = preds_inversed.tolist()
    days = getMonthdays()
    save_to_file(filename, days, y_preds)
    return send_file("btcMonthPrediction.csv", mimetype='application/x-csv', attachment_filename='btcMonthPrediction.csv', as_attachment=True)


@app.route("/ETHexport")
def exportETH():
    filename = "ethMonthPrediction.csv"
    preds_inversed = predict("KRW-ETH")
    y_preds = preds_inversed.tolist()
    days = getMonthdays()
    save_to_file(filename, days, y_preds)
    return send_file("ethMonthPrediction.csv", mimetype='application/x-csv', attachment_filename='ethMonthPrediction.csv', as_attachment=True)


app.run()
