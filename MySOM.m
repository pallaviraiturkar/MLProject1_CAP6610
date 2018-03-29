function [ClusterIm, CCIm] = MySOM(Im, ImType, NumClusts)
% Self-organizing Map clustering for RGB / Hyperspectral Image Clustering
% 
% Syntax: [ClusterIm, CCIm] = MyClust(Im, ImType, NumClusts)
% 
% Inputs:
%     Im - [input image]
%     ImType - [string argument] - image type - ('RGB','Hyper')
%     NumClusts - [positive integer] - number of clusters
%     
% Outputs:
%     ClusterIm - [double Mat] - labeled image with dimension row x col
%     CCIm - [double Mat] - tensor of binary image representing connected
%     components in ClusterIm (for RGB images ONLY)
% 
% Author: Pallavi A Raiturkar
% University of Florida, Dept of Computer and Information Science and Engineering
%%

if (strcmp(ImType,'RGB'))
    
    %resize image to speed up computation
    I = imresize(Im,0.5);
    %transform color space
    cform = makecform('srgb2lab');
    lab_I = applycform(I,cform);
    % lab_I = I;
    I = double(lab_I(:,:,2:3));
    nrows = size(I,1);
    ncols = size(I,2);
    
    %reshape into features
    I = reshape(I,nrows*ncols,2);
    
    I_1 = I(:,1);
    I_2 = I(:,2);
    
    %normalize vectors
    normA = (I_1-min(I_1(:))) ./ (max(I_1(:))-min(I_1(:)));
    normB = (I_2-min(I_2(:))) ./ (max(I_2(:))-min(I_2(:)));
    I = [normA normB];
    temp = size(I,1);

    % Max number of iteration
    N =90;
    % initial learning rate
    Eta = 0.3;
    % exponential decay rate of the learning rate
    Etadecay = 0.05;
    %random weight

    w = rand(2,NumClusts);
    %initial D
    D = zeros(1,NumClusts);
    % initial cluster index
    clusterindex = zeros(temp,1);
    % repeat for number of iterations 
    for t = 1:N
        for data = 1 : temp
            for c = 1 : NumClusts
                D(c) = sqrt(((w(1,c)-I(data,1))^2) + ((w(2,c)-I(data,2))^2));
            end
        %find best matching unit
        [~, bmuindex] = min(D);
        clusterindex(data)=bmuindex;

        %update weight
        oldW = w(:,bmuindex);
        new = oldW +  Eta * (reshape(I(data,:),2,1)-oldW);
        w(:,bmuindex) = new;

        end
        % update learning rate
        Eta= Etadecay * Eta;
    end

    %label each pixel in the image using the results from KMeans
    ClusterIm = reshape(clusterindex,nrows,ncols);
    
    % find connected components in each cluster
    CCIm = zeros(NumClusts, nrows * ncols); 
    
    for i = 1:NumClusts
        index = clusterindex == i;
        CCIm(i, :) = index * 1;
    end

elseif (strcmp(ImType, 'Hyper'))
    
    %resize image to speed up computation
    I = imresize(Im,0.5);
    lab_I = I;
    I = double(lab_I(:,:,2:3));
    nrows = size(I,1);
    ncols = size(I,2);
    
    %reshape into features
    I = reshape(I,nrows*ncols,2);
    
    I_1 = I(:,1);
    I_2 = I(:,2);
    
    %normalize vectors
    normA = (I_1-min(I_1(:))) ./ (max(I_1(:))-min(I_1(:)));
    normB = (I_2-min(I_2(:))) ./ (max(I_2(:))-min(I_2(:)));
    I = [normA normB];
    temp = size(I,1);

    % Max number of iteration
    N =90;
    % initial learning rate
    Eta = 0.3;
    % exponential decay rate of the learning rate
    Etadecay = 0.05;
    %random weight

    w = rand(2,NumClusts);
    %initial D
    D = zeros(1,NumClusts);
    % initial cluster index
    clusterindex = zeros(temp,1);
    % repeat for number of iterations 
    for t = 1:N
        for data = 1 : temp
            for c = 1 : NumClusts
                D(c) = sqrt(((w(1,c)-I(data,1))^2) + ((w(2,c)-I(data,2))^2));
            end
        %find best matching unit
        [~, bmuindex] = min(D);
        clusterindex(data)=bmuindex;

        %update weight
        oldW = w(:,bmuindex);
        new = oldW +  Eta * (reshape(I(data,:),2,1)-oldW);
        w(:,bmuindex) = new;

        end
        % update learning rate
        Eta= Etadecay * Eta;
    end

    %label each pixel in the image using the results from KMeans
    ClusterIm = reshape(clusterindex,nrows,ncols);
    
    % find connected components in each cluster
    CCIm = zeros(NumClusts, nrows * ncols); 
    
    for i = 1:NumClusts
        index = clusterindex == i;
        CCIm(i, :) = index * 1;
    end
    
    CCIm = []; % connected components are not required for hyperspectral images
    
end

end