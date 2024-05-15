library(forecast)
library(zoo)

setwd("C:/Users/srivi/Downloads/school stuff/MSBABAN 673 TSA")

# Loading dataset
sales.data <- read.csv("Food&Beverage.csv")

# Check for a few records in the dataset
head(sales.data)
tail(sales.data)

# Create time series data set sales.ts
sales.ts <- ts(sales.data$Value, 
							 start = c(1992, 1), end = c(2022, 12), freq = 12)
sales.ts

# Finding the mean / average (level component) of the data
mean(sales.ts)

# Plotting the time series data 
sales.lin <- tslm(sales.ts ~ trend)

plot(sales.ts, 
		 xlab = "Time", ylab = "Sales (in million dollars)", 
		 ylim = c(25000, 90000), xaxt = 'n',
		 main = "Sales of Food & Beverages Stores")
lines(sales.lin$fitted, lwd = 2, col = "blue")
axis(1, at = seq(1992, 2024, 1), labels = format(seq(1992, 2024, 1)))


sales.stl <- stl(sales.ts, s.window = "periodic")
autoplot(sales.stl, main = "Sales Time Series Components")

# Identify possible time series components.
autocor <- Acf(sales.ts, lag.max = 12, 
							 main = "Autocorrelation for Sales of Food & Beverages Stores")

# Create 24 months of validation data and remaining as training data
nValid <- 24
nTrain <- length(sales.ts) - nValid 
train.ts <- window(sales.ts, start = c(1992, 1), end = c(1992, nTrain))
valid.ts <- window(sales.ts, start = c(1992, nTrain + 1), 
									 end = c(1992, nTrain + nValid))

# Display the training and validation data
nTrain
nValid
train.ts
valid.ts

# Test predictability of the dataset

# Use Arima() function to fit AR(1) model.
# The ARIMA model of order = c(1,0,0) gives an AR(1) model.
sales.ar1<- Arima(sales.ts, order = c(1,0,0))
summary(sales.ar1)

# Apply z-test to test the null hypothesis that beta 
# coefficient of AR(1) is equal to 1.
ar1 <- 0.8826
s.e. <- 0.0256
null_mean <- 1
alpha <- 0.05
z.stat <- (ar1-null_mean)/s.e.
z.stat
p.value <- pnorm(z.stat)
p.value
if (p.value<alpha) {
	"Reject null hypothesis"
} else {
	"Accept null hypothesis"
}

# Create first differenced Amtrak Ridership data using lag1.
diff.sales.ts <- diff(sales.ts, lag = 1)
diff.sales.ts

# Use Acf() function to identify autocorrelation for first differenced 
# Amtrak Ridership, and plot autocorrelation for different lags 
# (up to maximum of 12).
Acf(diff.sales.ts, lag.max = 12, 
		main = "Autocorrelation for Differenced Food & Beverages Data")

# Applying forecasting methods to forecast sales in validation partition

# Forecasting Model 1: Two level forecast with linear trend and seasonality and trailing MA forecast with a window width of k = 4

# Develop using the tslm() function a regression model with linear trend and seasonality.
trend.seas <- tslm(train.ts ~ trend + season)
summary(trend.seas)

# Forecast monthly sales in the validation period with the forecast() function
trend.seas.pred <- forecast(trend.seas, h = nValid, level = 0)
trend.seas.pred

# Identify regression residuals in the training period
trend.seas.res <- trend.seas$residuals
trend.seas.res

##apply a trailing MA (window width of 4) for these residuals
ma.trail.res <- rollmean(trend.seas.res, k = 4, align = "right")
ma.trail.res

# Identify trailing MA forecast of these residuals in the validation period 
ma.trail.res.pred <- forecast(ma.trail.res, h = nValid, level = 0)
ma.trail.res.pred

# Develop two-level forecast for the validation period by combining the regression forecast and trailing MA forecast for residuals.
fst.2level <- trend.seas.pred$mean + ma.trail.res.pred$mean
fst.2level

# Creating a table that contains validation data, regression forecast, trailing MA forecast for residuals, and two-level (combined) forecast in the validation period.
valid.df <- round(data.frame(valid.ts, trend.seas.pred$mean, 
														 ma.trail.res.pred$mean, 
														 fst.2level), 3)
names(valid.df) <- c("Sales", "Regression.Fst", 
										 "MA.Residuals.Fst", "Combined.Fst")
valid.df

# Forecasting Model 2: Holt Winter's Model

# Develop a Holt-Winter’s (HW) model with automated selection of error, trend, and seasonality options, and automated selection of smoothing parameters for the training partition
hw.ZZZ <- ets(train.ts, model = "ZZZ")
hw.ZZZ

# Use the model to forecast monthly sales for the validation period 
hw.ZZZ.pred <- forecast(hw.ZZZ, h = nValid, level = 0)
hw.ZZZ.pred

# Forecasting Model 3: Quadratic trend and seasonality model

# FIT REGRESSION MODEL WITH QUADRATIC TREND AND SEASONALITY 
train.quad.season <- tslm(train.ts ~ trend + I(trend^2) + season)

# See summary of quadratic trend and seasonality model and associated parameters.
summary(train.quad.season)

# Apply forecast() function to make predictions for ts with trend and seasonality data in validation set.  
train.quad.season.pred <- forecast(train.quad.season, h = nValid, level = 0)
train.quad.season.pred

# Forecasting Model 4: Two Level Autoregressive Model

# Identify autocorrelation for the model residuals (training and validation sets), and plot autocorrelation for different lags (up to maximum of 12).
Acf(train.quad.season.pred$residuals, lag.max = 12, 
		main = "Autocorrelation for Food and Beverages Sales Training Residuals")
Acf(valid.ts - train.quad.season.pred$mean, lag.max = 12, 
		main = "Autocorrelation for Food and Beverages Sales Validation Residuals")

# Use Arima() function to fit AR(2) model for training residuals. The Arima model of order = c(2,0,0) gives an AR(2) model.
res.ar2 <- Arima(train.quad.season$residuals, order = c(2,0,0))
summary(res.ar2)
res.ar2$fitted

# Use Acf() function to identify autocorrelation for the training residual of residuals and plot autocorrelation for different lags 
Acf(res.ar2$residuals, lag.max = 12, 
		main = "Autocorrelation for Food and Beverages Training Residuals of Residuals")

# Use forecast() function to make prediction of residuals in validation set
res.ar2.pred <- forecast(res.ar2, h = nValid, level = 0)
res.ar2.pred

# Create two-level model's forecast with quadratic trend and seasonality regression + AR(2) for residuals for validation period.
valid.two.level.pred <- train.quad.season.pred$mean + res.ar2.pred$mean
valid.two.level.pred

# Create data table with validation data, regression forecast for validation period, AR(1) residuals for validation, and two level model results. 
valid.df <- round(data.frame(valid.ts, train.quad.season.pred$mean, 
														 res.ar2.pred$mean, valid.two.level.pred),3)
names(valid.df) <- c("Sales", "Reg.Forecast", 
										 "AR(2)Forecast", "Combined.Forecast")
valid.df


# Forecasting model 5: Auto Arima Model
# Use the auto.arima() function to develop an ARIMA model using the training data set.
train.auto.arima <- auto.arima(train.ts)
summary(train.auto.arima)

# Apply forecast() function to make predictions for ts with auto ARIMA model in validation set.  
train.auto.arima.pred <- forecast(train.auto.arima, h = nValid, level = 0)
train.auto.arima.pred

# Accuracy of all models in the validation data
round(accuracy(fst.2level, valid.ts), 3)
round(accuracy(hw.ZZZ.pred$mean, valid.ts), 3)
round(accuracy(train.quad.season.pred$mean, valid.ts),3)
round(accuracy(valid.two.level.pred, valid.ts), 3)
round(accuracy(train.auto.arima.pred$mean, valid.ts), 3)

# For entire dataset 

# Forecasting model 1: Two level forecast

# For the entire data set, identify the regression model with linear trend and seasonality 
tot.trend.seas <- tslm(sales.ts ~ trend  + season)

# Create regression forecast for future 24 periods.
tot.trend.seas.pred <- forecast(tot.trend.seas, h = 24, level = 0)
tot.trend.seas.pred

# Identify and display regression residuals for entire data set.
tot.trend.seas.res <- tot.trend.seas$residuals
tot.trend.seas.res

# Use trailing MA to forecast residuals for entire data set.
tot.ma.trail.res <- rollmean(tot.trend.seas.res, k = 4, align = "right")
tot.ma.trail.res

# Create forecast for trailing MA residuals for future 24 periods.
tot.ma.trail.res.pred <- forecast(tot.ma.trail.res, h = 24, level = 0)
tot.ma.trail.res.pred

# Develop 2-level forecast for future 12 periods by combining regression forecast and trailing MA for residuals for future 12 periods.
tot.fst.2level <- tot.trend.seas.pred$mean + tot.ma.trail.res.pred$mean
tot.fst.2level

# Forecasting Model 2: HW model

# Develop the HW model for the model with the automated selection of error, trend, and seasonality options, and automated selection of smoothing parameters.
HW.ZZZ <- ets(sales.ts, model = "ZZZ")
HW.ZZZ 

# Use the model to forecast monthly sales in the 12 months of 2023 and 2024 using the forecast() function
HW.ZZZ.pred <- forecast(HW.ZZZ, h = 24 , level = 0)
HW.ZZZ.pred

#Forecasting model 3: Quadratic Trend and Seasonality

# Create regression model with quadratic trend and seasonality
quad.season <- tslm(sales.ts ~ trend + I(trend^2) + season)

# See summary of quadratic trend and seasonality equation and associated parameters.
summary(quad.season)

# Apply forecast() function to make predictions for ts with quadratic trend and seasonality data in 24 future periods.
quad.season.pred <- forecast(quad.season, h = 24, level = 0)
quad.season.pred

# Forecasting Model 4: Two Level auto regressive Model

# Develop a two-level forecast (regression model with quadratic trend and seasonality and AR(2) model for residuals) for the entire data set

# Use tslm() function to create quadratic trend and seasonality model.
quad.season <- tslm(sales.ts ~ trend + I(trend^2) + season)

# See summary of quadratic trend equation and associated parameters.
summary(quad.season)

# Apply forecast() function to make predictions with quadratic trend and seasonal model into the future 24 periods.  
quad.season.pred <- forecast(quad.season, h = 24, level = 0)
quad.season.pred

# Make prediction of residuals into the future 24 months.
residual.ar2 <- Arima(quad.season$residuals, order = c(2,0,0))
residual.ar2.pred <- forecast(residual.ar2, h = 24, level = 0)

# Use summary() to identify parameters of AR(2) model.
summary(residual.ar2)

# Identify autocorrelation for the residuals of residuals and plot autocorrelation for different lags
Acf(residual.ar2$residuals, lag.max = 12, 
		main = "Autocorrelation for Residuals of Residuals for Entire Data Set")


# Identify forecast for the future 24 periods as sum of quadratic trend and seasonality model and AR(2) model for residuals.
quad.season.ar2.pred <- quad.season.pred$mean + residual.ar2.pred$mean
quad.season.ar2.pred


# Create a data table with quadratic trend and seasonal forecast for 24 future periods, AR(2) model for residuals for 8 future periods, and combined two-level forecast for 24 future periods. 
table.df <- round(data.frame(quad.season.pred$mean, 
														 residual.ar2.pred$mean, quad.season.ar2.pred),3)
names(table.df) <- c("Reg.Forecast", "AR(2)Forecast","Combined.Forecast")
table.df

#Forecasting Model 5: Auto Arima Model

# Use summary() to show auto ARIMA model and its parameters for entire data set.
auto.arima <- auto.arima(sales.ts)
summary(auto.arima)

# Apply forecast() function to make predictions for ts with auto ARIMA model for the future 24 periods. 
auto.arima.pred <- forecast(auto.arima, h = 24, level = 0)
auto.arima.pred

# Apply accuracy function to find accuracies for entire dataset of the models used
round(accuracy(tot.trend.seas.pred$fitted+tot.ma.trail.res, sales.ts), 3)
round(accuracy(HW.ZZZ.pred$fitted, sales.ts), 3)
round(accuracy(quad.season.pred$fitted, sales.ts),3)
round(accuracy(quad.season.pred$fitted + residual.ar2$fitted, sales.ts), 3)
round(accuracy(auto.arima.pred$fitted, sales.ts), 3)
round(accuracy((snaive(sales.ts))$fitted, sales.ts), 3)


# Create a table with regression forecast, trailing MA for residuals, and total forecast for future 12 periods.
future12.df <- round(data.frame(tot.trend.seas.pred$mean, tot.ma.trail.res.pred$mean, 
																tot.fst.2level), 3)
names(future12.df) <- c("Regression.Fst", "MA.Residuals.Fst", "Combined.Fst")
future12.df

# Plot original and forecasted Sales time series data for the final selected model
plot(sales.ts, 
		 xlab = "Time", ylab = "Sales", ylim = c(25000, 100000), 
		 bty = "l", xlim = c(1992, 2024.25), lwd =1, xaxt = "n",
		 main = "Sales Data and Regression with Trend and Seasonality") 
axis(1, at = seq(1992, 2024.25, 1), labels = format(seq(1992, 2024.25, 1)))
lines(tot.trend.seas.pred$fitted, col = "blue", lwd = 2)
lines(tot.trend.seas.pred$mean, col = "blue", lty =5, lwd = 2)
legend(1992,94000, legend = c("Sales", "Regression",
															"Regression Forecast for Future Periods"), 
			 col = c("black", "blue" , "blue"), 
			 lty = c(1, 1, 2), lwd =c(1, 2, 2), bty = "n")

# Plot on chart vertical lines and horizontal arrows describing
# entire data set and future prediction intervals.
lines(c(2022, 2022), c(0, 100000))
text(2005, 65000, "Training Data Set")
text(2023.1, 31000, "Future")
arrows(1992, 63000, 2022, 63000, code = 3, length = 0.1,
			 lwd = 1, angle = 30)
arrows(2022.1, 28000, 2024.1, 28000, code = 3, length = 0.1,
			 lwd = 1, angle = 30)
