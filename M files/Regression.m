hold on
test = csvread('data.solved.csv',1,0);
testX=test(:,1);
testY=test(:,2);
plot(testX,testY,'r');

train= csvread('data.xsq.train.csv',1,0);
trainX=train(:,1);
trainY=train(:,2);
scatter(trainX,trainY,4,'k');
hold off

