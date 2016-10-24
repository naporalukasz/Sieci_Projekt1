hold on
test = csvread('data.solved.csv',1,0);
testX=test(:,1);
testY=test(:,2);
testColor=test(:,3);
testColors=zeros(length(testColor),3);

for i=1:length(testColor)
  if(testColor(i)==1)
    testColors(i,:)=[1 0 0];
  end
   if(testColor(i)==2)
    testColors(i,:)=[0 1 0];
  end
   if(testColor(i)==3)
    testColors(i,:)=[0 0 1];
  end
  if(testColor(i)==4)
    testColors(i,:)=[0 1 1];
  end
end
scatter(testX,testY,10,testColors);



%train= csvread('data.circles.train.10000.csv',1,0);
%trainX=train(:,1);
%trainY=train(:,2);
%trainColor=train(:,3);
%trainColors=zeros(length(trainColor),3);
%
%for i=1:length(trainColor)
%  if(trainColor(i)==1)
%    trainColors(i,:)=[1 0 0];
%  end
%   if(trainColor(i)==2)
%    trainColors(i,:)=[0 1 0];
%  end
%   if(trainColor(i)==3)
%    trainColors(i,:)=[0 0 1];
%  end
%     if(trainColor(i)==4)
%    trainColors(i,:)=[0 1 1];
%  end
%end
%scatter(trainX,trainY,10,trainColors);
%hold off