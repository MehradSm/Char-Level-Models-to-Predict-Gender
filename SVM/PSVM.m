%% Data
clear,
clc
tol = 0.001;
N = 5000;
load('x_test.mat')
load('y_test.mat')
xtest = x_test(N+1:2*N,:);
ytest = y_test(N+1:2*N);
xtrain = x_test(1:N,:);
ytrain = y_test(1:N);

%% Transfering the data
x_train = zeros(N,58);
for i=2:59
    x_train(:,i-1) = sum(xtrain == i, 2);
end
x_test = zeros(N,58);
for i=2:59
    x_test(:,i-1) = sum(xtest == i, 2);
end

y_train = double(2*ytrain -1)';
y_test = double(2*ytest -1)';

%% Training
p = 1;
C = 25;
K_train = (x_train*x_train'+1).^p;
K_test = (x_test*x_train'+1).^p;
[alpha,bias] = smo(K_train, y_train', C, tol);

train_acc = mean(y_train.*(K_train*(alpha'.*y_train) + bias) > 0)
test_acc = mean(y_test.*(K_test*(alpha'.*y_train) + bias) > 0)

num_supp = sum(alpha>0)
