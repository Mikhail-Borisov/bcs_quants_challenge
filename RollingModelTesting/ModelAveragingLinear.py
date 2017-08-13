import pyflux as pf
model1 = pf.ARIMA(data=y_train.values, ar=1, ma=0)
model2 = pf.ARIMA(data=y_train.values, ar=1, ma=0)
model3 = pf.LLEV(data=y_train.values)
model5 = pf.GPNARX(data=y_train.values, ar=1, kernel=pf.OrnsteinUhlenbeck())
model6 = pf.GPNARX(data=y_train.values, ar=2, kernel=pf.SquaredExponential())
mix = pf.Aggregate(learning_rate=1.0, loss_type='squared')