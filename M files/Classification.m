M = csvread('data.solved.csv',1,0);
X=M(:,1);
Y=M(:,2);
col3=M(:,3);
%c = linspace(1,10,length(x));
scatter(X,Y,1,col3);