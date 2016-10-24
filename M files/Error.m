hold on
iterError = csvread('data.error.csv',1,0);
iters=iterError(:,1);
trainError=iterError(:,2);
testError=iterError(:,3);
%axis([0 5000 0.04 0.08]);
figure(1);
plot(iters,trainError,iters,testError);
hold off