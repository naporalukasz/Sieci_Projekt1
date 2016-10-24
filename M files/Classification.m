hold on
test = csvread('data.solved.csv',1,0);
testX=test(:,1);
testY=test(:,2);
color=test(:,3);
scatter(testX,testY,1,color);

train= csvread('data.train.csv',1,0);
trainX=train(:,1);
trainY=train(:,2);
color=train(:,3);
scatter(trainX,trainY,4,color);
hold off