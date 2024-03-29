clear,clc;
addpath('ClusteringMeasure','Datasets','Tools');

ds_list = {'3sources','100leaves','BBC','BBCSport','Caltech101_7','CUB'};

for ds_iter = 1:length(ds_list)
    ds = ds_list{ds_iter};
    clear X;
    
    current_time = datestr(now());
    fprintf('---------------------------------------------------------------------------------\n');

    dataname = ds;
    load(strcat(dataname,'.mat'));
    gt = Y;
    v = length(X);
    num_cls = length(unique(gt));
    num_sam = size(X{1},2);
    
    for k = 1:v
        X{k} = X{k}./(repmat(sqrt(sum(X{k}.^2,1)),size(X{k},1),1)+1e-8);
        X{k} = X{k}';
    end
    
    Z = cell(1,v);
    
    for k = 1:v
        Z{k} = getZ(X{k},X{k});
        Z{k} = full(Z{k}');
    end

    [C,pre_result] = FPGL_main(Z,gt);
    
    NMI = nmi(pre_result,gt);
    Purity = purity(gt, pre_result);
    ACC = Accuracy(pre_result,double(gt));
    [Fscore,Precision,~] = compute_f(gt,pre_result);
    [AR,RI,~,~]=RandIndex(gt,pre_result);
    fprintf('Dataset: %s\t NMI %.4f, ACC %.4f, Purity %.4f, Fscore %.4f, Precision %.4f \n',dataname,NMI,ACC,Purity,Fscore,Precision);
end


