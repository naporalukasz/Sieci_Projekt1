hold on

M = csvread('data.error.csv',1,0);

X=M(:,1);

trainError=M(:,2);

testError=M(:,3);

axis([0 5000 0.04 0.08]);

figure(1);

plot(X,trainError,X,testError);