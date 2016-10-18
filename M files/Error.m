M = csvread('data.error.csv',1,0);
X=M(:,1);
trainError=M(:,2);
testError=M(:,3);
plot(X,trainError,X,testError);