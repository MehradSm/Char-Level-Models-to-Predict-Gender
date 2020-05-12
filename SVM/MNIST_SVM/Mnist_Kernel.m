clear,
clc

%% Data
load('mnist_49_28x28.mat')
X_train = x(1:1000,:); y_train = y(1:1000);
X_val = x(1001:2000,:); y_val = y(1001:2000);
X_test = x(2001:end,:); y_test = y(2001:end);
tol = 0.0001;
gamma = 0.01;

%% Training
val_er = zeros(2,16);
for p = 1:2
    if p == 1
        K_train = (X_train*X_train'+1);
        K_val = (X_val*X_train'+1);
    else
        K_train = sum(X_train.^2,2)*ones(1,size(X_train,1)) ...
            -2*(X_train*X_train')+...
            (sum(X_train.^2,2)*ones(1,size(X_train,1)))';
        K_train = exp(-gamma*K_train);
        
        K_val = sum(X_val.^2,2)*ones(1,size(X_train,1)) ...
            -2*(X_val*X_train')+...
            (sum(X_train.^2,2)*ones(1,size(X_val,1)))';
        K_val = exp(-gamma*K_val);
    end
    for C = 400:4:400
        [alpha,bias] = smo(K_train, y_train', C, tol);
        val_er(p,C/4 + 1) = mean(y_val.*(K_val*(alpha'.*y_train) + bias) > 0);
    end
end
[val_er1,C1] = max(val_er(1,:));
[val_er2,C2] = max(val_er(2,:));


K_train1 = X_train*X_train'+1;
K_test1 = X_test*X_train'+1;
K_train2 = sum(X_train.^2,2)*ones(1,size(X_train,1)) ...
            -2*(X_train*X_train')+...
            (sum(X_train.^2,2)*ones(1,size(X_train,1)))';
        K_train2 = exp(-gamma*K_train2);
        
K_test2 = sum(X_test.^2,2)*ones(1,size(X_train,1)) ...
            -2*(X_test*X_train')+...
            (sum(X_train.^2,2)*ones(1,size(X_test,1)))';
K_test2 = exp(-gamma*K_test2);

[alpha1,bias1] = smo(K_train1, y_train', C1, tol);
[alpha2,bias2] = smo(K_train2, y_train', C2, tol);


%%
train_acc_Li = mean(y_train.*(K_train1*(alpha1'.*y_train) + bias1) > 0)
test_acc_Li = mean(y_test.*(K_test1*(alpha1'.*y_train) + bias1) > 0)

train_acc_RBF = mean(y_train.*(K_train2*(alpha2'.*y_train) + bias2) > 0)
test_acc_RBF = mean(y_test.*(K_test2*(alpha2'.*y_train) + bias2) > 0)


%% 
[v1, I1] = sort(y_train.*(K_train1*(alpha1'.*y_train) + bias1));
[v2, I2] = sort(y_train.*(K_train2*(alpha2'.*y_train) + bias2));
I = I1;
figure(1)
for p = 1:2
    for i = 1:20
        subplot(4,5,i)
        imagesc(reshape(X_train(I(i),:),28,28)')
        colormap(gray)
        axis square
        axis off
        if y(I(i)) == -1
            title('4')
        else
            title('9')
        end
    end
    figure(2)
    I = I2;
end


