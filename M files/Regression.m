M = csvread('data.solved.csv',1,0);
col1=M(:,1);
col2=M(:,2);
plot(col1,col2);